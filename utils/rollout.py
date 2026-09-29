"""Online experience collection (one env, one step at a time) and evaluation."""

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np


def example_transition(example_batch):
    """Row layout of the replay buffers."""
    return dict(
        observations=example_batch['observations'][0],
        actions=example_batch['actions'][0],
        rewards=np.float32(0.0),
        masks=np.float32(1.0),
        terminals=np.float32(0.0),
        empowerment=np.float32(0.0),
        is_offline=np.float32(0.0),
        distill_target=np.float32(0.0),
        distill_target_ready=np.float32(0.0),
    )


class Collector:
    """Steps the env with the stochastic actor, conditioned on the episode's goal, and writes rows to the buffer.

    With an empowerment estimator, rows get E(s) = nan until `flush_empowerment` fills everything added since the last
    flush in one batched call; the bonus target (max E over the episode, incl. its final observation) is written once
    the episode has ended. With RND, `flush_rnd` feeds the visited states (in order) to the running RND stats.
    """

    RND_CHUNK = 128

    def __init__(self, env, buffer, seed):
        self.env = env
        self.buffer = buffer
        self.rng = jax.random.PRNGKey(seed)
        self._pending_empowerment = []  # abs indices with E(s) still nan
        self._pending_rnd = []  # abs indices not yet fed to the RND stats
        self._pending_rnd_first = []  # is the row an episode's reset state (not any transition's s')?
        self._closed_episodes = []  # (first row, marker row) of episodes closed since the last flush
        self._episode_start_abs = None
        self._reset_episode()

    def _reset_episode(self):
        self.observation, info = self.env.reset()
        self.goal = info['goal']
        self.ep_return, self.ep_length, self.ep_success = 0.0, 0, 0.0
        self._episode_first_row = True

    def flush_empowerment(self, agent):
        if not self._pending_empowerment:
            return
        abs_idxs = np.asarray(self._pending_empowerment, dtype=np.int64)
        self._pending_empowerment = []
        self.rng, key = jax.random.split(self.rng)
        values = agent.empowerment_np(self.buffer.read_field('observations', abs_idxs), key)
        self.buffer.write_field('empowerment', abs_idxs, values)

        closed, self._closed_episodes = self._closed_episodes, []
        for start_abs, end_abs in closed:
            start_abs = max(start_abs, self.buffer.oldest_abs)
            if start_abs >= end_abs:
                continue
            rows = np.arange(start_abs, end_abs + 1)
            values = self.buffer.read_field('empowerment', rows)
            self.buffer.write_field('distill_target', rows[:-1], np.full(len(rows) - 1, values.max(), np.float32))
            self.buffer.write_field('distill_target_ready', rows[:-1], np.ones(len(rows) - 1, dtype=np.float32))

    def flush_rnd(self, agent):
        """Returns `(agent, info)` with the RND stats updated on every state visited since the last call."""
        if not self._pending_rnd:
            return agent, {}
        abs_idxs = np.asarray(self._pending_rnd, dtype=np.int64)
        is_next = 1.0 - np.asarray(self._pending_rnd_first, dtype=np.float32)
        self._pending_rnd, self._pending_rnd_first = [], []
        observations = self.buffer.read_field('observations', abs_idxs)

        info = {}
        chunk = self.RND_CHUNK
        for start in range(0, len(abs_idxs), chunk):
            obs = observations[start : start + chunk]
            nxt = is_next[start : start + chunk]
            valid = np.ones((len(obs),), dtype=np.float32)
            pad = chunk - len(obs)
            if pad > 0:  # pad to one jit shape
                obs = np.concatenate([obs, np.zeros((pad,) + obs.shape[1:], dtype=obs.dtype)], axis=0)
                nxt = np.concatenate([nxt, np.zeros((pad,), dtype=np.float32)], axis=0)
                valid = np.concatenate([valid, np.zeros((pad,), dtype=np.float32)], axis=0)
            agent, info = agent.rnd_update_stats(jnp.asarray(obs), jnp.asarray(nxt), jnp.asarray(valid))
        return agent, {k: float(np.asarray(v)) for k, v in info.items()}

    def step(self, agent):
        """One env step. Returns the episode stats if the episode ended, else None."""
        self.rng, key = jax.random.split(self.rng)
        action = np.asarray(agent.sample_actions(observations=self.observation, goals=self.goal, seed=key))
        action = np.clip(action, -1, 1)

        next_observation, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        self.ep_return += float(reward)
        self.ep_length += 1
        self.ep_success = float(info.get('success', self.ep_success))

        nan_if = lambda flag: np.float32(np.nan if flag else 0.0)  # noqa: E731
        abs_idx = self.buffer.add_transition(
            dict(
                observations=self.observation,
                actions=action,
                rewards=np.float32(reward),
                masks=np.float32(1.0 - float(terminated)),
                terminals=np.float32(done),
                empowerment=nan_if(agent.uses_empowerment),
                is_offline=np.float32(0.0),
                distill_target=nan_if(agent.uses_empowerment),
                distill_target_ready=np.float32(0.0),
            )
        )
        if agent.uses_empowerment:
            if self._episode_start_abs is None:
                self._episode_start_abs = abs_idx
            self._pending_empowerment.append(abs_idx)
        if agent.uses_rnd:
            self._pending_rnd.append(abs_idx)
            self._pending_rnd_first.append(self._episode_first_row)
        self._episode_first_row = False
        self.observation = next_observation

        if not done:
            return None
        end_abs = self.buffer.end_trajectory(next_observation)
        if agent.uses_empowerment:
            # the final observation also needs its E (it's in the episode max)
            self.buffer.write_field('empowerment', [end_abs], np.array([np.nan], dtype=np.float32))
            self._pending_empowerment.append(end_abs)
            self._closed_episodes.append((self._episode_start_abs, end_abs))
            self._episode_start_abs = None
        if agent.uses_rnd:
            self._pending_rnd.append(end_abs)
            self._pending_rnd_first.append(False)
        episode = dict(episode_return=self.ep_return, episode_length=self.ep_length, episode_success=self.ep_success)
        self._reset_episode()
        return episode


def evaluate(agent, env, num_episodes, temperature=0.0):
    """Random-task episodes with the deterministic actor; returns mean success / return / length."""
    rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
    stats = defaultdict(list)
    for _ in range(num_episodes):
        observation, info = env.reset()
        goal = info['goal']
        done, step, ep_return = False, 0, 0.0
        while not done:
            rng, key = jax.random.split(rng)
            action = np.clip(
                np.array(agent.sample_actions(observation, goal, seed=key, temperature=temperature)), -1, 1
            )
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            ep_return += float(reward)
        stats['success'].append(float(info.get('success', 0.0)))
        stats['episode_return'].append(ep_return)
        stats['episode_length'].append(step)
    return {k: float(np.mean(v)) for k, v in stats.items()}
