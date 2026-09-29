import os

import numpy as np
import ogbench

import envs  # noqa: F401  (registers the online envs)


def _dataset_dir_kwargs():
    # OGBENCH_DATASET_DIR overrides ogbench's default ~/.ogbench/data (handy on clusters)
    if os.environ.get('OGBENCH_DATASET_DIR'):
        return {'dataset_dir': os.environ['OGBENCH_DATASET_DIR']}
    return {}


def load_datasets(dataset_name):
    """Compact (observation-only) train / val datasets of an ogbench dataset, as raw dicts."""
    return ogbench.make_env_and_datasets(dataset_name, compact_dataset=True, dataset_only=True, **_dataset_dir_kwargs())


def make_online_env(env_name, episode_length=None):
    """The env for an online run, e.g. `antmaze-medium-corner-all-squares-online-v0` (see envs/__init__.py)."""
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    if episode_length is not None:
        # overwrite the registered TimeLimit
        e = env
        while e is not None:
            if getattr(e, '_max_episode_steps', None) is not None:
                e._max_episode_steps = int(episode_length)
            e = getattr(e, 'env', None)
        if env.spec is not None:
            env.spec.max_episode_steps = int(episode_length)
    env.reset()
    return env


def env_horizon(env):
    spec = getattr(env, 'spec', None) or getattr(env.unwrapped, 'spec', None)
    return None if spec is None else spec.max_episode_steps


def make_example_batch(env):
    """Shape-only example batch for `agent.create`."""
    return dict(
        observations=np.zeros((1, *env.observation_space.shape), dtype=np.float32),
        actions=np.zeros((1, *env.action_space.shape), dtype=np.float32),
    )
