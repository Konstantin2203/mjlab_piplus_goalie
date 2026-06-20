#!/usr/bin/env -S uv run python
"""Run `git pull` in the project dir on every machine in an experiment queue.

Reads the same YAML as run_experiments.py, dedupes targets by (host, dir) so
each checkout is pulled once even if a host has multiple GPUs, and runs
`git -C <project_dir> pull` locally or over ssh.

Usage:
  uv run scripts/sync_targets.py experiments.example.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_experiments import Config, load_config


def pull(cfg: Config, host: str, project_dir: str) -> bool:
  git = ["git", "-C", project_dir, "pull", "--ff-only"]
  if host in {"local", "localhost", ""}:
    label = f"local:{project_dir}"
    cmd = git
  else:
    label = f"{host}:{project_dir}"
    cmd = ["ssh", *cfg.ssh_opts, host, " ".join(git)]
  print(f"\n=== {label} ===")
  res = subprocess.run(cmd, capture_output=True, text=True)
  out = (res.stdout + res.stderr).strip()
  if out:
    print(out)
  ok = res.returncode == 0
  print("OK" if ok else f"FAILED (rc={res.returncode})")
  return ok


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("queue_file", type=Path, help="YAML experiment queue file")
  args = ap.parse_args()

  cfg = load_config(args.queue_file, only=None)

  seen: set[tuple[str, str]] = set()
  results = []
  for t in cfg.targets:
    project_dir = t.project_dir or cfg.project_dir
    key = (t.host, project_dir)
    if key in seen:
      continue
    seen.add(key)
    results.append((f"{t.host}:{project_dir}", pull(cfg, t.host, project_dir)))

  print("\n==== summary ====")
  for label, ok in results:
    print(f"  {'✓' if ok else '✗'} {label}")
  if not all(ok for _, ok in results):
    sys.exit(1)


if __name__ == "__main__":
  main()
