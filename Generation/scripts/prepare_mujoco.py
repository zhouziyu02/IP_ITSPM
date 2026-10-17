"""Regenerate real HopperPhysics trajectories using the Diff-MN data recipe.

Source: TimeCraft, Diff-MN/datasets/mujoco_physics.py (Microsoft MIT license).
This uses the real Hopper physics model; no synthetic proxy is substituted.
Regenerated data are explicitly labelled and versioned; they are not claimed
to be a bit-for-bit reconstruction of another dataset release.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'Generation/itspm_generation/table1_data')
    p.add_argument('--lengths', type=int, nargs='+', default=[36])
    p.add_argument('--n-trajectories', type=int, default=4620,
                   help='4620 is the published Diff-MN HopperPhysics default.')
    p.add_argument('--data-seed', type=int, default=123)
    args = p.parse_args()
    if min(args.lengths) < 2 or args.n_trajectories < 10:
        p.error('Need sequence lengths >=2 and at least 10 trajectories.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in args.lengths if not (args.output_dir / f'mujoco_training_{n}.pt').exists()]
    if not missing:
        print('Requested MuJoCo datasets already exist; preserving them.', flush=True)
        return
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from dm_control import suite
    env = suite.load('hopper', 'stand')
    physics = env.physics
    if physics.data.qpos.size != 7 or physics.data.qvel.size != 7:
        raise RuntimeError('Unexpected Hopper model dimensions: expected 7 positions + 7 velocities.')
    rng = np.random.RandomState(args.data_seed)
    data = np.zeros((args.n_trajectories, max(missing), 14), dtype=np.float64)
    started = time.monotonic()
    for i in range(args.n_trajectories):
        with physics.reset_context():
            physics.data.qpos[:2] = rng.uniform(0, 0.5, size=2)
            physics.data.qpos[2:] = rng.uniform(-2, 2, size=physics.data.qpos[2:].shape)
            physics.data.qvel[:] = rng.uniform(-5, 5, size=physics.data.qvel.shape)
        for t in range(max(missing)):
            data[i, t, :7] = physics.data.qpos
            data[i, t, 7:] = physics.data.qvel
            physics.step()
        if (i + 1) % 250 == 0 or i + 1 == args.n_trajectories:
            print(f'HopperPhysics {i + 1}/{args.n_trajectories} ({time.monotonic() - started:.1f}s)', flush=True)
    if not np.isfinite(data).all():
        raise RuntimeError('Physics generated non-finite values; refusing to save.')
    for length in missing:
        path = args.output_dir / f'mujoco_training_{length}.pt'
        torch.save(torch.from_numpy(data[:, :length].copy()), path)
        metadata = {
            'dataset': 'MuJoCo HopperPhysics', 'provenance': 'regenerated_physical_simulation',
            'source': 'TimeCraft: Diff-MN/datasets/mujoco_physics.py',
            'source_license': 'MIT', 'n_trajectories': args.n_trajectories,
            'seq_len': length, 'channels': 14, 'data_seed': args.data_seed,
            'dm_control_version': importlib.metadata.version('dm-control'),
            'mujoco_version': importlib.metadata.version('mujoco'),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'note': 'Real HopperPhysics trajectories generated with the recorded versions and seed; not a claim of bit-identical original data.',
        }
        path.with_suffix('.provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Saved {path} shape={list(data[:, :length].shape)}', flush=True)


if __name__ == '__main__':
    main()
