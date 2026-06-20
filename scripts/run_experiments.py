#!/usr/bin/env -S uv run python
"""Run a queue of training experiments across machines/GPUs (work-stealing).

A YAML file defines a list of `experiments` (each a name + train CLI overrides)
and a static list of `targets` (host:gpu). One worker thread is started per
target; each worker pulls the next experiment from a shared queue, runs it to
completion on its target, then pulls the next. Faster targets naturally process
more experiments (work-stealing), so the queue drains as fast as the hardware
allows.

Remote (ssh) experiments run inside a detached tmux session on the target, so a
dropped ssh connection does NOT stop the run. The orchestrator tracks progress
with short, fresh ssh polls; if polling can't reach the host it just retries,
and the training keeps going in tmux meanwhile (requires tmux on each target).

Usage:
  uv run scripts/run_experiments.py experiments.yaml
  uv run scripts/run_experiments.py experiments.yaml --dry-run
  uv run scripts/run_experiments.py experiments.yaml --only baseline,high_push

YAML schema (see experiments.example.yaml):

  task: Mjlab-Velocity-PiPlus      # registered task id
  project_dir: /homes/15guelden/mjlab_piplus  # default; per-target overridable
  common_args:                     # overrides applied to every experiment
    env.scene.num-envs: 4096
  targets:                         # static host:gpu list; host "local" == no ssh
    - {host: local,  gpu: 0}
    - {host: tams01, gpu: 0, project_dir: /data/me/mjlab_piplus}
    - {host: tams01, gpu: 1, project_dir: /data/me/mjlab_piplus}
  experiments:
    - name: baseline
      args: {}
    - name: high_push
      args:
        env.events.push_robot.params.velocity_range.x: "[-1.5, 1.5]"

Override values map to tyro CLI flags: each `key: value` becomes
`--key value` (booleans -> True/False, lists -> tuple literal "(a, b)", which
is the form tyro's sequence fields accept). The `--gpu-ids` flag is set
automatically per target and must not appear in args.
"""

from __future__ import annotations

import argparse
import base64
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

LOCAL_HOSTS = {"local", "localhost", ""}


@dataclass
class Target:
  host: str
  gpu: int
  # Path to the project on THIS host. Falls back to Config.project_dir when None.
  project_dir: str | None = None

  @property
  def is_local(self) -> bool:
    return self.host in LOCAL_HOSTS

  def __str__(self) -> str:
    return f"{self.host or 'local'}:{self.gpu}"


@dataclass
class Experiment:
  name: str
  args: dict

  # Filled in as the experiment runs.
  # pending | waiting (for GPU) | running | done | failed | detached
  status: str = "pending"
  target: str = ""
  returncode: int | None = None
  run_name: str = ""
  log_path: Path | None = None
  duration_s: float = 0.0


@dataclass
class Config:
  task: str
  # Default project path, used for any target that doesn't set its own.
  project_dir: str | None
  common_args: dict
  targets: list[Target]
  experiments: list[Experiment]
  ssh_opts: list[str] = field(default_factory=list)
  # How often (seconds) to poll a remote tmux session for completion, and how
  # many consecutive failed polls to tolerate before giving up contact (the run
  # keeps going in tmux regardless — see _run_remote).
  poll_interval: float = 15.0
  max_poll_failures: int = 40
  # A target's GPU is considered free to use when it has at least
  # min_free_mem_mb free and utilization <= max_util_pct. Workers wait,
  # re-checking every gpu_check_interval seconds, until their GPU is free.
  min_free_mem_mb: int = 16000
  max_util_pct: int = 30
  gpu_check_interval: float = 30.0


EXIT_MARKER = "__MJLAB_EXIT__"
ALIVE_MARKER = "__MJLAB_ALIVE__"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: Path, only: set[str] | None) -> Config:
  raw = yaml.safe_load(path.read_text())
  if not isinstance(raw, dict):
    sys.exit(f"{path}: top-level YAML must be a mapping")

  task = raw.get("task")
  if not task:
    sys.exit(f"{path}: missing required key 'task'")
  default_dir = raw.get("project_dir")

  targets = []
  for t in raw.get("targets", []):
    host = str(t.get("host", "local"))
    proj = t.get("project_dir", default_dir)
    if not proj:
      sys.exit(
        f"{path}: target {host}:{t.get('gpu')} has no project_dir "
        "(set it on the target or provide a top-level 'project_dir')"
      )
    targets.append(Target(host=host, gpu=int(t["gpu"]), project_dir=str(proj)))
  if not targets:
    sys.exit(f"{path}: at least one target is required")

  experiments = []
  for e in raw.get("experiments", []):
    name = e["name"]
    if only and name not in only:
      continue
    experiments.append(Experiment(name=name, args=dict(e.get("args") or {})))
  if not experiments:
    sys.exit(f"{path}: no experiments to run (after --only filter)")

  return Config(
    task=task,
    project_dir=default_dir,
    common_args=dict(raw.get("common_args") or {}),
    targets=targets,
    experiments=experiments,
    ssh_opts=list(raw.get("ssh_opts") or []),
    poll_interval=float(raw.get("poll_interval", 15.0)),
    max_poll_failures=int(raw.get("max_poll_failures", 40)),
    min_free_mem_mb=int(raw.get("min_free_mem_mb", 16000)),
    max_util_pct=int(raw.get("max_util_pct", 30)),
    gpu_check_interval=float(raw.get("gpu_check_interval", 30.0)),
  )


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def _format_value(value) -> str:
  """Render a YAML value as a single tyro CLI argv token.

  Sequences become Python tuple literals, e.g. "(-0.25, 0.25)" — this is the
  form tyro accepts for its sequence fields (a bracketed "[..]" or
  space-separated values are NOT parsed correctly for them).
  """
  if isinstance(value, bool):
    return "True" if value else "False"
  if isinstance(value, (list, tuple)):
    inner = ", ".join(str(v) for v in value)
    if len(value) == 1:
      inner += ","  # keep it a tuple literal, not a parenthesized scalar
    return f"({inner})"
  return str(value)


def _slug(name: str) -> str:
  """Turn an experiment name into a wandb-friendly run-name slug."""
  return "-".join(name.lower().replace("_", " ").split())


def build_train_argv(cfg: Config, exp: Experiment, target: Target) -> list[str]:
  merged = {**cfg.common_args, **exp.args}
  if "gpu-ids" in merged or "gpu_ids" in merged:
    sys.exit(
      f"experiment {exp.name!r}: do not set 'gpu-ids' in args; "
      "it is assigned per target"
    )
  # Name the wandb run after the experiment. mjlab renders --agent.run-name as
  # "<timestamp>_<run_name>" and uses it as the wandb display name, e.g.
  # 2026-06-18_20-39-58_lower-in-the-knees. Let an explicit override win.
  if "agent.run-name" not in merged and "agent.run_name" not in merged:
    merged["agent.run-name"] = _slug(exp.name)
  argv = ["uv", "run", "train", cfg.task, "--gpu-ids", f"[{target.gpu}]"]
  for key, value in merged.items():
    argv += [f"--{key}", _format_value(value)]
  return argv


def _remote_session(exp: Experiment, target: Target, run_ts: str) -> str:
  """tmux session name unique per experiment+gpu+launch.

  tmux rejects names containing '.' or ':', so the experiment slug (which can
  contain dots, e.g. 'push-1.0') is sanitized to dashes here.
  """
  raw = f"mjlab-{_slug(exp.name)}-g{target.gpu}-{run_ts}"
  return raw.replace(".", "-").replace(":", "-")


def _train_str(cfg: Config, exp: Experiment, target: Target) -> str:
  argv = build_train_argv(cfg, exp, target)
  project_dir = target.project_dir or cfg.project_dir
  return "cd {} && {}".format(
    shlex.quote(project_dir),
    " ".join(shlex.quote(tok) for tok in argv),
  )


def build_remote_payload(
  cfg: Config, exp: Experiment, target: Target, remote_log: str
) -> str:
  """Shell script (run by `zsh -l` on the remote) that runs training, tees
  output to remote_log, and records the exit code so we can read it later."""
  q = shlex.quote
  logdir = q(str(Path(remote_log).parent))
  rlog = q(remote_log)
  # uv installs to ~/.local/bin, which is only on PATH for interactive zsh
  # (.zshrc); non-interactive login shells skip it, so add it explicitly.
  return (
    'export PATH="$HOME/.local/bin:$PATH"; '
    f"mkdir -p {logdir}; "
    f"{{ {_train_str(cfg, exp, target)} ; }} > {rlog} 2>&1; "
    f"echo {EXIT_MARKER}$? >> {rlog}"
  )


def build_remote_launch(
  cfg: Config, exp: Experiment, target: Target, session: str, remote_log: str
) -> list[str]:
  """ssh argv that starts the run in a detached tmux session.

  The run survives our SSH connection dropping: tmux keeps it alive on the
  remote host, and we reconnect with fresh `ssh` calls to poll for completion.
  The payload is base64-encoded to sidestep multi-layer shell quoting.
  """
  payload = build_remote_payload(cfg, exp, target, remote_log)
  b64 = base64.b64encode(payload.encode()).decode()
  # Decoded and executed by a login shell so `uv`/CUDA are on PATH.
  inner = f"printf %s {b64} | base64 -d | zsh -l"
  remote = (
    f"tmux kill-session -t {shlex.quote(session)} 2>/dev/null; "
    f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(inner)}"
  )
  return ["ssh", *cfg.ssh_opts, target.host, remote]


def build_command(cfg: Config, exp: Experiment, target: Target) -> list[str]:
  """Return a subprocess argv that runs the experiment on `target`.

  Used for actual local launches and for --dry-run display. Remote launches go
  through build_remote_launch (with a per-run session name) instead.
  """
  if target.is_local:
    return ["zsh", "-lc", _train_str(cfg, exp, target)]
  session = _remote_session(exp, target, "<ts>")
  return build_remote_launch(cfg, exp, target, session, f"/tmp/{session}.log")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def log(target: Target | None, msg: str) -> None:
  ts = datetime.now().strftime("%H:%M:%S")
  prefix = f"[{ts}]" + (f" [{target}]" if target else "")
  with _print_lock:
    print(f"{prefix} {msg}", flush=True)


def _extract_run_name(log_path: Path, task: str) -> str:
  """Best-effort scrape of the rsl_rl run name from a captured log."""
  needle = f"logs/rsl_rl/{task.lower()}"
  try:
    for line in log_path.read_text(errors="replace").splitlines():
      idx = line.find("logs/rsl_rl/")
      if idx == -1:
        continue
      path_part = line[idx:].split()[0].rstrip(",'\"")
      if needle in path_part.lower() or "logs/rsl_rl/" in path_part:
        return Path(path_part).name
  except OSError:
    pass
  return ""


def _parse_exit_code(log_path: Path) -> int | None:
  """Read the EXIT_MARKER line our remote payload appends to the log."""
  try:
    for line in reversed(log_path.read_text(errors="replace").splitlines()):
      if line.startswith(EXIT_MARKER):
        try:
          return int(line[len(EXIT_MARKER) :].strip())
        except ValueError:
          return None
  except OSError:
    pass
  return None


def _run_local(
  cfg: Config, exp: Experiment, target: Target, log_path: Path, stop: threading.Event
) -> None:
  cmd = build_command(cfg, exp, target)
  with log_path.open("w") as fh:
    fh.write(f"# {exp.name} on {target}\n# {' '.join(cmd)}\n\n")
    fh.flush()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    while True:
      try:
        proc.wait(timeout=2.0)
        break
      except subprocess.TimeoutExpired:
        if stop.is_set():
          proc.terminate()
          try:
            proc.wait(timeout=10)
          except subprocess.TimeoutExpired:
            proc.kill()
          break
  exp.returncode = proc.returncode


def _kill_remote(cfg: Config, target: Target, session: str) -> None:
  subprocess.run(
    ["ssh", *cfg.ssh_opts, target.host, f"tmux kill-session -t {shlex.quote(session)}"],
    capture_output=True,
    text=True,
  )


def _poll_remote(
  cfg: Config, target: Target, session: str, remote_log: str
) -> tuple[bool, bool, str]:
  """Probe a remote run with one fresh ssh call.

  Returns (ok, alive, log_text). ok=False means the ssh probe itself failed
  (e.g. dropped connection) — the run is unaffected and we just retry.
  """
  q = shlex.quote
  probe = (
    f"cat {q(remote_log)} 2>/dev/null; "
    f"printf '\\n%s' {ALIVE_MARKER}; "
    f"tmux has-session -t {q(session)} >/dev/null 2>&1 && printf 1 || printf 0"
  )
  res = subprocess.run(
    ["ssh", *cfg.ssh_opts, target.host, probe], capture_output=True, text=True
  )
  idx = res.stdout.rfind(ALIVE_MARKER)
  if res.returncode != 0 and idx == -1:
    return False, False, ""
  if idx == -1:
    return False, False, res.stdout
  alive = res.stdout[idx + len(ALIVE_MARKER) :].strip() == "1"
  return True, alive, res.stdout[:idx]


def _run_remote(
  cfg: Config, exp: Experiment, target: Target, log_path: Path, stop: threading.Event
) -> str:
  """Launch in detached tmux and poll. Returns an outcome string:
  'finished' | 'requeue' | 'detached' | 'stopped'.
  """
  run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
  session = _remote_session(exp, target, run_ts)
  remote_log = f"/tmp/{session}.log"
  launch = build_remote_launch(cfg, exp, target, session, remote_log)

  res = subprocess.run(launch, capture_output=True, text=True)
  if res.returncode != 0:
    log_path.write_text(
      f"# launch failed (rc={res.returncode}) on {target}\n"
      f"# {' '.join(launch)}\n\n{res.stdout}\n{res.stderr}\n"
    )
    # rc 255 == ssh could not connect -> host unreachable, hand work back.
    return "requeue" if res.returncode == 255 else "finished"
  log(target, f"tmux session {session} started for {exp.name}")

  fails = 0
  while True:
    if stop.wait(cfg.poll_interval):
      _kill_remote(cfg, target, session)
      return "stopped"
    ok, alive, text = _poll_remote(cfg, target, session, remote_log)
    if not ok:
      fails += 1
      if fails >= cfg.max_poll_failures:
        log(
          target,
          f"lost contact with {target.host} after {fails} polls; {exp.name} "
          f"left running in tmux session {session} (attach to check on it)",
        )
        return "detached"
      log(target, f"poll failed ({fails}/{cfg.max_poll_failures}); retrying")
      continue
    fails = 0
    if text:
      log_path.write_text(text)
    if not alive:
      exp.returncode = _parse_exit_code(log_path)
      return "finished"


def _query_gpu(cfg: Config, target: Target) -> tuple[int, int] | None:
  """Return (free_mem_mb, util_pct) for target's GPU, or None if unavailable."""
  query = [
    "nvidia-smi",
    "--query-gpu=memory.free,utilization.gpu",
    "--format=csv,noheader,nounits",
    "-i",
    str(target.gpu),
  ]
  cmd = (
    query if target.is_local else ["ssh", *cfg.ssh_opts, target.host, " ".join(query)]
  )
  res = subprocess.run(cmd, capture_output=True, text=True)
  if res.returncode != 0:
    return None
  try:
    free_s, util_s = res.stdout.strip().split(",")
    return int(free_s), int(util_s)
  except ValueError:
    return None


def _wait_for_gpu(
  cfg: Config, exp: Experiment, target: Target, stop: threading.Event
) -> bool:
  """Block until target's GPU is free enough to use. Returns False if `stop`
  was set while waiting (caller should abort)."""
  announced = False
  while not stop.is_set():
    stats = _query_gpu(cfg, target)
    if stats is None:
      log(target, f"could not query GPU for {exp.name}; will retry")
    else:
      free_mb, util = stats
      if free_mb >= cfg.min_free_mem_mb and util <= cfg.max_util_pct:
        return True
      if not announced:
        log(
          target,
          f"GPU busy (free={free_mb}MB, util={util}%); waiting for "
          f">={cfg.min_free_mem_mb}MB free and <={cfg.max_util_pct}% util "
          f"before starting {exp.name}",
        )
        announced = True
    if stop.wait(cfg.gpu_check_interval):
      return False
  return False


def run_experiment(
  cfg: Config,
  exp: Experiment,
  target: Target,
  log_dir: Path,
  stop: threading.Event,
) -> str:
  """Run one experiment on one target. Returns an outcome string the worker
  uses to decide whether to re-queue or retire."""
  log_path = log_dir / f"{exp.name}.log"
  exp.log_path = log_path
  exp.target = str(target)

  # Don't launch until the assigned GPU is actually free to use.
  exp.status = "waiting"
  if not _wait_for_gpu(cfg, exp, target, stop):
    exp.status = "stopped"
    return "stopped"

  exp.status = "running"
  log(target, f"START {exp.name}  -> {log_path}")

  start = time.time()
  if target.is_local:
    _run_local(cfg, exp, target, log_path, stop)
    outcome = "finished"
  else:
    outcome = _run_remote(cfg, exp, target, log_path, stop)
  exp.duration_s = time.time() - start
  exp.run_name = _extract_run_name(log_path, cfg.task)

  if outcome == "requeue":
    log(target, f"SSH FAILED for {exp.name} (rc=255) — re-queuing experiment")
  elif outcome == "detached":
    exp.status = "detached"
  elif outcome == "stopped":
    exp.status = "failed"
    log(target, f"STOPPED {exp.name}")
  elif exp.returncode == 0:
    exp.status = "done"
    log(target, f"DONE  {exp.name}  ({exp.duration_s / 60:.1f} min)  {exp.run_name}")
  else:
    exp.status = "failed"
    log(target, f"FAIL  {exp.name}  rc={exp.returncode}  (see {log_path})")
  return outcome


def worker(
  cfg: Config,
  target: Target,
  q: "queue.Queue[Experiment]",
  log_dir: Path,
  stop: threading.Event,
) -> None:
  while not stop.is_set():
    try:
      exp = q.get_nowait()
    except queue.Empty:
      return
    try:
      outcome = run_experiment(cfg, exp, target, log_dir, stop)
      if outcome == "requeue":
        # Host unreachable: hand the experiment back and retire this worker so
        # we don't spin on a dead host.
        exp.status = "pending"
        exp.target = ""
        q.put(exp)
        log(target, "retiring worker (host unreachable)")
        return
      if outcome == "detached":
        # Run is still alive in tmux but we can't track it; retire this worker
        # without re-queuing (re-running would double up on a busy GPU).
        log(target, "retiring worker (lost contact, run left detached)")
        return
    finally:
      q.task_done()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("queue_file", type=Path, help="YAML experiment queue file")
  ap.add_argument(
    "--only",
    default="",
    help="comma-separated experiment names to run (default: all)",
  )
  ap.add_argument(
    "--dry-run",
    action="store_true",
    help="print the command for each experiment/target and exit",
  )
  ap.add_argument(
    "--log-dir",
    type=Path,
    default=None,
    help="where to write per-experiment logs (default: experiment_runs/<ts>)",
  )
  args = ap.parse_args()

  only = {s for s in args.only.split(",") if s} or None
  cfg = load_config(args.queue_file, only)

  if args.dry_run:
    print(f"task={cfg.task}")
    for t in cfg.targets:
      print(f"  target {t}  dir={t.project_dir or cfg.project_dir}")
    # Show the launch on the first remote target if any (it exercises tmux),
    # otherwise the first target.
    example = next((t for t in cfg.targets if not t.is_local), cfg.targets[0])
    for exp in cfg.experiments:
      print(f"\n=== {exp.name} (example on {example}) ===")
      print("  " + " ".join(shlex.quote(t) for t in build_command(cfg, exp, example)))
    return

  log_dir = args.log_dir or Path(
    "experiment_runs", datetime.now().strftime("%Y%m%d_%H%M%S")
  )
  log_dir.mkdir(parents=True, exist_ok=True)

  q: "queue.Queue[Experiment]" = queue.Queue()
  for exp in cfg.experiments:
    q.put(exp)

  stop = threading.Event()
  log(
    None,
    f"{len(cfg.experiments)} experiments across {len(cfg.targets)} targets "
    f"-> logs in {log_dir}",
  )

  threads = [
    threading.Thread(target=worker, args=(cfg, t, q, log_dir, stop), daemon=True)
    for t in cfg.targets
  ]
  for th in threads:
    th.start()

  try:
    while any(th.is_alive() for th in threads):
      for th in threads:
        th.join(timeout=0.5)
  except KeyboardInterrupt:
    log(None, "interrupted — terminating running experiments...")
    stop.set()
    for th in threads:
      th.join()

  # Summary.
  log(None, "==== summary ====")
  by_status: dict[str, int] = {}
  for exp in cfg.experiments:
    by_status[exp.status] = by_status.get(exp.status, 0) + 1
    mark = {
      "done": "✓",
      "failed": "✗",
      "running": "…",
      "waiting": "⏸",
      "pending": "·",
      "detached": "⇢",
    }.get(exp.status, "?")
    log(
      None,
      f"  {mark} {exp.name:30s} {exp.status:8s} {exp.target:14s} "
      f"{exp.duration_s / 60:5.1f}min  {exp.run_name}",
    )
  log(None, "  " + "  ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
  if by_status.get("failed") or by_status.get("pending") or by_status.get("detached"):
    sys.exit(1)


if __name__ == "__main__":
  main()
