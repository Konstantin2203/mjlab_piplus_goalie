"""Sanity check: applying the PiPlus symmetry twice must recover the original.

The symmetry mirror is an involution (mirror∘mirror = identity). Since
``PiPlusSymmetry.__call__`` prepends the original batch to the mirrored batch
(num_aug = 2), the batch size doubles on every call:

    call 1 (N -> 2N):  [orig, M(orig)]
    call 2 (2N -> 4N): [orig, M(orig), M(orig), M(M(orig))]

So after two calls, with quarters q0..q3 of size N each:
    q0 == orig
    q1 == M(orig) == q2
    q3 == M(M(orig)) == orig == q0   <-- the involution check
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab_piplus.tasks.mdp.symmetry import PiPlusSymmetry

N = 8
OBS_DIM = 47  # 3 ang_vel + 3 proj_grav + 12 jpos + 12 jvel + 12 act + 2 phase + 3 command
ACT_DIM = 12

torch.manual_seed(0)

actor = torch.randn(N, OBS_DIM)
obs = TensorDict({"actor": actor.clone()}, batch_size=[N])
actions = torch.randn(N, ACT_DIM)

sym = PiPlusSymmetry()

# First application: N -> 2N
obs1, act1 = sym(None, obs, actions)
assert obs1.batch_size[0] == 2 * N, obs1.batch_size
assert act1.shape[0] == 2 * N, act1.shape

# Second application: 2N -> 4N
obs2, act2 = sym(None, obs1, act1)
assert obs2.batch_size[0] == 4 * N, obs2.batch_size
assert act2.shape[0] == 4 * N, act2.shape


def report(name: str, x: torch.Tensor) -> None:
    q = x.shape[0] // 4
    q0, q1, q2, q3 = x[:q], x[q : 2 * q], x[2 * q : 3 * q], x[3 * q :]

    # involution: q3 (mirror of mirror) must equal q0 (orig)
    invol = (q3 - q0).abs().max().item()
    # consistency: q1 and q2 are both a single mirror of orig
    mirror_consistent = (q1 - q2).abs().max().item()

    print(f"\n[{name}]")
    print(f"  q0 (orig)            sample: {q0[0, :6].tolist()}")
    print(f"  q1 (mirror)          sample: {q1[0, :6].tolist()}")
    print(f"  involution err (q3 vs q0)   : {invol:.3e}")
    print(f"  mirror consistency (q1 vs q2): {mirror_consistent:.3e}")
    ok = invol < 1e-6 and mirror_consistent < 1e-6
    # also confirm a mirror actually changes the data (not a no-op)
    changed = (q0 - q1).abs().max().item()
    print(f"  mirror actually changes data : {changed:.3e} (should be > 0)")
    print(f"  RESULT: {'PASS' if ok and changed > 0 else 'FAIL'}")


report("obs.actor", obs2["actor"])
report("actions", act2)
