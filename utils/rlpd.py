"""RLPD: the offline dataset goes into a second trajectory buffer, and every batch takes an exact `offline_ratio`
share of its rows from it (goals are relabeled the same way as for online rows).

Offline rows carry `is_offline = 1`, and, when the agent has an empowerment estimator, E(s) of every row (computed
once and cached next to the estimator checkpoint) and the empowerment bonus target max_k E(s_k) over the row's
trajectory.
"""

import dataclasses
import os
import time

import jax
import numpy as np

from utils.env_utils import load_datasets
from utils.replay_buffer import TrajectoryReplayBuffer


@dataclasses.dataclass
class BufferSource:
    buffer: TrajectoryReplayBuffer
    discount: float

    def sample(self, batch_size):
        return self.buffer.sample(batch_size, self.discount)


class MixedBatchSampler:
    """Exact-ratio mix of an online and an offline source."""

    def __init__(self, online, offline, offline_ratio):
        assert 0.0 < offline_ratio < 1.0
        self.online = online
        self.offline = offline
        self.offline_ratio = float(offline_ratio)

    def split(self, batch_size):
        num_offline = int(round(batch_size * self.offline_ratio))
        num_offline = min(max(num_offline, 1), batch_size - 1)
        return batch_size - num_offline, num_offline

    def sample(self, batch_size):
        num_online, num_offline = self.split(batch_size)
        online_batch = self.online.sample(num_online)
        offline_batch = self.offline.sample(num_offline)
        return {key: np.concatenate([online_batch[key], offline_batch[key]], axis=0) for key in online_batch}


def offline_trajectories(dataset):
    """(first_row, marker_row) per trajectory; ogbench compact datasets mark each trajectory's last state valids=0."""
    markers = np.flatnonzero(~(np.asarray(dataset['valids']) > 0))
    starts = np.concatenate([[0], markers[:-1] + 1])
    for start, marker in zip(starts, markers):
        if marker > start:
            yield int(start), int(marker)


def offline_empowerment_values(agent, observations, dataset_name, seed=0):
    """E(s) of every dataset row, cached in <estimator run dir>/empowerment_values/."""
    config = agent.config
    cache_dir = os.path.join(config['emp_checkpoint_path'], 'empowerment_values')
    cache_path = os.path.join(
        cache_dir,
        f'{dataset_name}_e{config["emp_restore_epoch"]}_n{int(config["emp_num_splus_samples"])}_seed{seed}.npy',
    )
    if os.path.exists(cache_path):
        values = np.load(cache_path)
        if values.shape == (len(observations),):
            print(f'[rlpd] loaded offline empowerment from {cache_path}')
            return values.astype(np.float32)

    start = time.time()
    values = agent.empowerment_np(observations, jax.random.PRNGKey(int(seed))).astype(np.float32)
    print(f'[rlpd] offline empowerment of {len(values)} rows in {time.time() - start:.0f}s (mean {values.mean():.4f})')
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f'{cache_path}.tmp.{os.getpid()}'
    with open(tmp_path, 'wb') as f:
        np.save(f, values)
    os.replace(tmp_path, cache_path)
    return values


def make_offline_source(dataset_name, agent, example_transition, seed=0):
    """Offline `BufferSource`, one row per env step."""
    dataset, _ = load_datasets(dataset_name)
    observations = np.asarray(dataset['observations'])
    actions = np.asarray(dataset['actions'])
    terminals = np.asarray(dataset['terminals'], dtype=np.float32)
    size = len(observations)
    assert observations.shape[1:] == np.asarray(example_transition['observations']).shape

    empowerment = np.zeros((size,), dtype=np.float32)
    distill = np.zeros((size,), dtype=np.float32)
    if agent.uses_empowerment:
        empowerment = offline_empowerment_values(agent, observations, dataset_name, seed=seed)
        for start, marker in offline_trajectories(dataset):
            distill[start : marker + 1] = empowerment[start : marker + 1].max()

    buffer = TrajectoryReplayBuffer.create(example_transition, size)
    for start, marker in offline_trajectories(dataset):
        for t in range(start, marker):
            buffer.add_transition(
                dict(
                    observations=observations[t],
                    actions=actions[t],
                    rewards=np.float32(0.0),
                    masks=np.float32(1.0 - terminals[t]),
                    terminals=terminals[t],
                    empowerment=empowerment[t],
                    is_offline=np.float32(1.0),
                    distill_target=distill[t],
                    distill_target_ready=np.float32(1.0),
                )
            )
        end_abs = buffer.end_trajectory(observations[marker])
        buffer.write_field('empowerment', [end_abs], np.asarray([empowerment[marker]]))
    print(f'[rlpd] offline dataset {dataset_name}: {buffer.num_valid} rows')
    return BufferSource(buffer, discount=float(agent.config['discount']))
