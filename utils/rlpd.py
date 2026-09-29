"""RLPD: the offline dataset goes into a second trajectory buffer, and every batch takes an exact `offline_ratio`
share of its rows from it (goals are relabeled the same way as for online rows).

Offline rows carry `is_offline = 1`, and, when the agent has an empowerment estimator, E(s) of every row (computed
once and cached next to the estimator checkpoint) and the empowerment bonus target max_k E(s_k) over the row's
trajectory.

For SUPE there is one offline row per env step t: (s_t, u_t, s_{t+k}), u_t the frozen OPAL posterior's label of the
window [t, t + k). Only windows inside one trajectory are anchors (SUPE `ChunkDataset`); the rest are stored as
non-anchor rows so trajectories stay contiguous.
"""

import dataclasses
import os
import time

import jax
import numpy as np

from utils.datasets import Dataset, SequenceDataset
from utils.env_utils import load_datasets
from utils.replay_buffer import TrajectoryReplayBuffer


@dataclasses.dataclass
class BufferSource:
    buffer: TrajectoryReplayBuffer
    discount: float
    next_offset: int = 1

    def sample(self, batch_size):
        return self.buffer.sample(batch_size, self.discount, next_offset=self.next_offset)


@dataclasses.dataclass
class GoalBankSource(BufferSource):
    """Goal-conditioned SUPE: every sampled offline row gets a task goal drawn uniformly from `goal_bank`."""

    goal_bank: np.ndarray = None

    def sample(self, batch_size):
        batch = super().sample(batch_size)
        batch['task_goals'] = self.goal_bank[np.random.randint(len(self.goal_bank), size=batch_size)]
        return batch


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


def label_offline_windows(agent, dataset, k, seed=0):
    """SUPE: u = tanh(posterior mean of q(z | window)) for the window [t, t + k) of every row t -> [size, D]."""
    seq_dataset = SequenceDataset(Dataset.create(**dataset), dict(sequence_length=k))
    rng = jax.random.PRNGKey(seed)
    labels = []
    for obs_seq, act_seq, seq_mask in seq_dataset.iter_windows(int(agent.config['label_chunk_bytes'])):
        rng, block_rng = jax.random.split(rng)
        labels.append(np.asarray(jax.device_get(agent.label_chunk_skills(obs_seq, act_seq, seq_mask, block_rng))))
    return np.concatenate(labels, axis=0)


def full_window_mask(dataset, k):
    """Rows t whose window [t, t + k) lies inside one trajectory."""
    keep = np.zeros((len(dataset['observations']),), dtype=bool)
    for start, marker in offline_trajectories(dataset):
        keep[start : max(start, marker - k + 1)] = True
    return keep


def make_offline_source(dataset_name, agent, example_transition, seed=0):
    """Offline `BufferSource` with one row per env step (flat CRL rows, or SUPE macro rows)."""
    dataset, _ = load_datasets(dataset_name)
    observations = np.asarray(dataset['observations'])
    actions = np.asarray(dataset['actions'])
    terminals = np.asarray(dataset['terminals'], dtype=np.float32)
    size = len(observations)
    assert observations.shape[1:] == np.asarray(example_transition['observations']).shape

    macro = agent.config['agent_name'] == 'supe'
    k, keep = 1, None
    if macro:
        k = int(agent.config['skill_commitment_k'])
        actions = label_offline_windows(agent, dataset, k, seed=seed)
        if agent.config['offline_full_windows_only']:
            keep = full_window_mask(dataset, k)
    extra = {}
    if 'task_goals' in example_transition:
        extra['task_goals'] = np.zeros_like(example_transition['task_goals'])  # drawn at sample time

    empowerment = np.zeros((size,), dtype=np.float32)
    distill = np.zeros((size,), dtype=np.float32)
    if agent.uses_empowerment:
        empowerment = offline_empowerment_values(agent, observations, dataset_name, seed=seed)
        for start, marker in offline_trajectories(dataset):
            distill[start : marker + 1] = empowerment[start : marker + 1].max()

    action_dtype = np.asarray(example_transition['actions']).dtype
    buffer = TrajectoryReplayBuffer.create(example_transition, size)
    for start, marker in offline_trajectories(dataset):
        for t in range(start, marker):
            last = min(t + k - 1, marker - 1)  # last env step of the (macro) transition
            buffer.add_transition(
                dict(
                    **extra,
                    observations=observations[t],
                    actions=np.asarray(actions[t], dtype=action_dtype),
                    rewards=np.float32(0.0),
                    masks=np.float32(1.0 - terminals[last]),
                    terminals=terminals[last],
                    empowerment=empowerment[t],
                    is_offline=np.float32(1.0),
                    distill_target=distill[t],
                    distill_target_ready=np.float32(1.0),
                ),
                valid=True if keep is None else bool(keep[t]),
            )
        end_abs = buffer.end_trajectory(observations[marker])
        buffer.write_field('empowerment', [end_abs], np.asarray([empowerment[marker]]))
    print(f'[rlpd] offline dataset {dataset_name}: {buffer.num_valid} rows')
    return BufferSource(buffer, discount=float(agent.config['discount']), next_offset=k)
