import dataclasses
from typing import Any

import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """Dataset class (supports ogbench's compact datasets, where next_observations come from shifting by one).

    For compact datasets, `valids` masks out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


@dataclasses.dataclass
class GCDataset:
    """Samples transitions together with a future-state s+ (`value_goals`).

    s+ is either a future state of the same trajectory at a geometric offset (clamped to the trajectory's last valid
    state), with prob `value_p_trajgoal`, or a uniformly random valid dataset state, with prob `value_p_randomgoal`.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        assert self.terminal_locs[-1] == self.size - 1
        assert np.isclose(self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0)

    def sample(self, batch_size):
        idxs = self.dataset.get_random_idxs(batch_size)
        batch = self.dataset.sample(batch_size, idxs)
        batch['value_goals'] = self.dataset['observations'][self.sample_goals(idxs)]
        return batch

    def sample_goals(self, idxs):
        batch_size = len(idxs)
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
        traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)

        return np.where(np.random.rand(batch_size) < self.config['value_p_trajgoal'], traj_goal_idxs, random_goal_idxs)
