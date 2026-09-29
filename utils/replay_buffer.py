"""Trajectory-aware ring buffer for online goal-conditioned RL.

Every env step stores one row; when an episode ends its final observation is appended as an extra marker row that is
never sampled as an anchor (like `valids` in ogbench's compact datasets), so `observations[i + 1]` is always the next
observation. Goals are relabeled at sample time: a future observation of the same trajectory with
P(offset = j) proportional to discount^j over the rows up to the trajectory end (in-progress trajectories count up to
the latest row). The same goal is used for the critic and the actor.

Rows are indexed by absolute step count, a row is held iff total - capacity <= abs_idx < total.
"""

import jax
import numpy as np


def _truncated_geometric_offsets(max_offsets, discount):
    """Offsets j in [1, d] with P(j) proportional to discount^j (inverse cdf), per row."""
    d = np.asarray(max_offsets, dtype=np.int64)
    assert np.all(d >= 1)
    if discount <= 0.0:
        return np.ones_like(d)
    u = np.random.rand(d.shape[0])
    offsets = np.floor(np.log1p(-u * (1.0 - discount**d)) / np.log(discount)).astype(np.int64) + 1
    return np.clip(offsets, 1, d)


class TrajectoryReplayBuffer:
    @classmethod
    def create(cls, example_transition, capacity):
        """Allocate from an example unbatched transition dict."""

        def alloc(example):
            example = np.asarray(example)
            return np.zeros((capacity, *example.shape), dtype=example.dtype)

        return cls(jax.tree_util.tree_map(alloc, dict(example_transition)))

    def __init__(self, data):
        self._data = data
        self.capacity = len(data['observations'])
        self.total = 0  # rows ever written (real + marker)
        self._valids = np.zeros((self.capacity,), dtype=np.bool_)
        self._num_valid = 0
        self._valid_slots_cache = (-1, None)
        self._traj_end = np.full((self.capacity,), -1, dtype=np.int64)  # abs idx of the trajectory's marker row
        self._open_rows = []  # abs indices of the in-progress trajectory

    @property
    def size(self):
        return min(self.total, self.capacity)

    @property
    def num_valid(self):
        return self._num_valid

    @property
    def oldest_abs(self):
        return max(0, self.total - self.capacity)

    def _slot(self, abs_idx):
        return abs_idx % self.capacity

    def _write_row(self, transition, valid):
        slot = self._slot(self.total)
        for key, buffer in self._data.items():
            buffer[slot] = transition[key]
        if self.total >= self.capacity and self._valids[slot]:
            self._num_valid -= 1
        self._valids[slot] = valid
        self._num_valid += int(valid)
        self._traj_end[slot] = -1
        self.total += 1
        return self.total - 1

    def add_transition(self, transition):
        """Append one transition of the in-progress trajectory, returns its absolute index."""
        abs_idx = self._write_row(transition, valid=True)
        self._open_rows.append(abs_idx)
        return abs_idx

    def end_trajectory(self, final_observation):
        """Close the trajectory with a marker row holding its final observation, returns the marker's abs index."""
        marker = {key: np.zeros_like(buffer[0]) for key, buffer in self._data.items()}
        marker['observations'] = final_observation
        end_abs = self._write_row(marker, valid=False)
        for abs_idx in self._open_rows + [end_abs]:
            if abs_idx >= self.oldest_abs:
                self._traj_end[self._slot(abs_idx)] = end_abs
        self._open_rows = []
        return end_abs

    def read_field(self, key, abs_idxs):
        abs_idxs = np.asarray(abs_idxs, dtype=np.int64)
        assert np.all(abs_idxs >= self.oldest_abs) and np.all(abs_idxs < self.total)
        return self._data[key][self._slot(abs_idxs)]

    def write_field(self, key, abs_idxs, values):
        abs_idxs = np.asarray(abs_idxs, dtype=np.int64)
        assert np.all(abs_idxs >= self.oldest_abs) and np.all(abs_idxs < self.total)
        self._data[key][self._slot(abs_idxs)] = values

    def valid_field(self, key):
        """`key` over every sampleable row currently held."""
        return self._data[key][np.flatnonzero(self._valids[: self.size])]

    def _valid_anchor_slots(self):
        # valid rows that already have a successor (the newest row of an open trajectory doesn't)
        if self._valid_slots_cache[0] != self.total:
            slots = np.flatnonzero(self._valids[: self.size])
            slots = slots[slots != self._slot(self.total - 1)]
            self._valid_slots_cache = (self.total, slots)
        return self._valid_slots_cache[1]

    def sample(self, batch_size, discount):
        """Batch of stored fields + `next_observations`, `value_goals` / `actor_goals` (the same future obs)."""
        valid_slots = self._valid_anchor_slots()
        oldest = self.oldest_abs
        latest = self.total - 1

        slots = valid_slots[np.random.randint(len(valid_slots), size=batch_size)]
        base = oldest - (oldest % self.capacity)
        abs_idxs = base + slots
        abs_idxs = np.where(abs_idxs < oldest, abs_idxs + self.capacity, abs_idxs)

        ends = self._traj_end[slots]
        ends = np.where(ends < 0, latest, ends)
        goal_abs = abs_idxs + _truncated_geometric_offsets(ends - abs_idxs, discount)
        next_abs = np.minimum(abs_idxs + 1, ends)

        batch = {key: buffer[slots] for key, buffer in self._data.items()}
        batch['next_observations'] = self._data['observations'][self._slot(next_abs)]
        goals = self._data['observations'][self._slot(goal_abs)]
        batch['value_goals'] = goals
        batch['actor_goals'] = goals
        return batch
