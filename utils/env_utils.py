import os
import pickle
import zlib

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


def _env_chain(env):
    chain = [env]
    while hasattr(chain[-1], 'env'):
        chain.append(chain[-1].env)
    return chain


def get_env_state(env):
    """Everything needed to continue `env` exactly (for resuming online runs), as compressed pickled bytes: the
    attributes of every wrapper and of the base env, incl. its MjData and np_random. Attributes that don't survive a
    pickle round trip (cameras, scene options, cube's dm_control mjcf scene) are fixed at construction and skipped;
    `set_env_state` keeps the fresh env's ones. The maze envs also draw tasks and teleport exits from the global
    np.random, which the caller has to save itself."""
    state = []
    for e in _env_chain(env):
        attrs = {}
        for key, value in vars(e).items():
            if key == 'env':
                continue
            try:
                pickle.loads(pickle.dumps(value))
            except Exception:  # noqa: BLE001, S112  (anything that can't be restored is construction-time)
                continue
            attrs[key] = value
        state.append((type(e).__name__, attrs))
    return zlib.compress(pickle.dumps(state), 1)  # MjData / MjModel are mostly empty arena


def set_env_state(env, state):
    """Load `get_env_state` output into a freshly made env of the same kind."""
    state = pickle.loads(zlib.decompress(state))
    chain = _env_chain(env)
    assert [type(e).__name__ for e in chain] == [name for name, _ in state], 'env wrapper chain changed'
    for e, (_, attrs) in zip(chain, state):
        vars(e).update(attrs)


def env_horizon(env):
    spec = getattr(env, 'spec', None) or getattr(env.unwrapped, 'spec', None)
    return None if spec is None else spec.max_episode_steps


def make_example_batch(env):
    """Shape-only example batch for `agent.create`."""
    return dict(
        observations=np.zeros((1, *env.observation_space.shape), dtype=np.float32),
        actions=np.zeros((1, *env.action_space.shape), dtype=np.float32),
    )
