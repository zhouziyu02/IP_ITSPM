"""Install and verify the official Hopper environment on Colab Python 3.13.

labmaze is only used by maze tasks, is not imported by the Hopper suite, and
has no Python 3.13 wheel. Install the unmodified official dm-control package
with all other declared dependencies, then verify real Hopper dynamics.
"""
import importlib.metadata
import os
import subprocess
import sys


def main():
    deps = ['absl-py>=0.7.0', 'dm-env', 'dm-tree!=0.1.2', 'glfw', 'lxml',
            'mujoco==3.13.0', 'numpy', 'protobuf>=3.19.4', 'PyOpenGL>=3.1.4',
            'pyparsing>=3', 'requests', 'setuptools!=50.0.0', 'scipy', 'tqdm']
    subprocess.run([sys.executable, '-m', 'pip', 'install', *deps], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', 'dm-control==1.0.46'], check=True)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import numpy as np
    from dm_control import suite
    env = suite.load('hopper', 'stand')
    physics = env.physics
    if (physics.data.qpos.size, physics.data.qvel.size) != (7, 7):
        raise RuntimeError('Hopper dimensions differ from the reference data recipe.')
    rng = np.random.RandomState(123)
    with physics.reset_context():
        physics.data.qpos[:2] = rng.uniform(0, .5, size=2)
        physics.data.qpos[2:] = rng.uniform(-2, 2, size=5)
        physics.data.qvel[:] = rng.uniform(-5, 5, size=7)
    for _ in range(36):
        physics.step()
    if not np.isfinite(physics.get_state()).all():
        raise RuntimeError('Hopper physics smoke run produced nonfinite state.')
    print('OFFICIAL_HOPPER_PHYSICS_VERIFIED',
          'dm-control=' + importlib.metadata.version('dm-control'),
          'mujoco=' + importlib.metadata.version('mujoco'), flush=True)


if __name__ == '__main__':
    main()
