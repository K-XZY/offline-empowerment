"""Online experience collection (one env, one step at a time) and evaluation."""

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np


def example_transition(example_batch, agent):
    """Row layout of the replay buffers (SUPE stores its latent skill u as the action)."""
    example_skill = getattr(agent, 'example_skill', None)
    row = dict(
        observations=example_batch['observations'][0],
        actions=example_batch['actions'][0] if example_skill is None else example_skill(),
        rewards=np.float32(0.0),
        masks=np.float32(1.0),
        terminals=np.float32(0.0),
        empowerment=np.float32(0.0),
        is_offline=np.float32(0.0),
        distill_target=np.float32(0.0),
        distill_target_ready=np.float32(0.0),
    )
    if getattr(agent, 'stores_task_goals', False):
        row['task_goals'] = np.zeros_like(example_batch['observations'][0])
    return row


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
        """One env step. Returns (env steps taken, episode stats if the episode ended else None)."""
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
            return 1, None
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
        return 1, episode


class MacroCollector(Collector):
    """SUPE: one row per macro step. The high level picks u (the OPAL prior during warm-up), the frozen decoder runs
    it for k env steps (or until the episode ends), and the row is (s_t, u, R, mask) with
    R = sum_i gamma_low^i (r_i + reward_shift)."""

    def __init__(self, env, buffer, seed, k, gamma_low, reward_shift, store_task_goals):
        self.k = int(k)
        self.gamma_low = float(gamma_low)
        self.reward_shift = float(reward_shift)
        self.store_task_goals = store_task_goals
        self.env_steps = 0
        self.last_transition = None  # newest (s, u), for SUPE's per-macro-step RND update
        super().__init__(env, buffer, seed)

    def step(self, agent):
        self.rng, skill_key, low_key = jax.random.split(self.rng, 3)
        skill = agent.sample_skills(
            observations=self.observation, goals=self.goal, seed=skill_key, temperature=1.0, env_steps=self.env_steps
        )
        skill = np.asarray(skill, dtype=np.float32)

        start_observation = self.observation
        macro_return, disc = 0.0, 1.0
        terminated_any, done = False, False
        env_steps = 0
        for _ in range(self.k):
            low_key, action_key = jax.random.split(low_key)
            action = np.asarray(agent.low_level_actions(observations=self.observation, skills=skill, seed=action_key))
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            env_steps += 1
            self.ep_return += float(reward)
            self.ep_length += 1
            self.ep_success = float(info.get('success', self.ep_success))
            macro_return += disc * (float(reward) + self.reward_shift)
            disc *= self.gamma_low
            terminated_any = terminated_any or bool(terminated)
            self.observation = next_observation
            done = bool(terminated or truncated)
            if done:
                break
        self.env_steps += env_steps

        nan_if = lambda flag: np.float32(np.nan if flag else 0.0)  # noqa: E731
        row = dict(
            observations=start_observation,
            actions=skill,
            rewards=np.float32(macro_return),
            masks=np.float32(1.0 - float(terminated_any)),
            terminals=np.float32(done),
            empowerment=nan_if(agent.uses_empowerment),
            is_offline=np.float32(0.0),
            distill_target=nan_if(agent.uses_empowerment),
            distill_target_ready=np.float32(0.0),
        )
        if self.store_task_goals:
            row['task_goals'] = self.goal  # fixed for the whole episode
        abs_idx = self.buffer.add_transition(row)
        self.last_transition = dict(observations=start_observation, actions=skill)
        if agent.uses_empowerment:
            if self._episode_start_abs is None:
                self._episode_start_abs = abs_idx
            self._pending_empowerment.append(abs_idx)

        if not done:
            return env_steps, None
        end_abs = self.buffer.end_trajectory(self.observation)
        if agent.uses_empowerment:
            self.buffer.write_field('empowerment', [end_abs], np.array([np.nan], dtype=np.float32))
            self._pending_empowerment.append(end_abs)
            self._closed_episodes.append((self._episode_start_abs, end_abs))
            self._episode_start_abs = None
        episode = dict(episode_return=self.ep_return, episode_length=self.ep_length, episode_success=self.ep_success)
        self._reset_episode()
        return env_steps, episode


def evaluate(agent, env, num_episodes, temperature=0.0):
    """Random-task episodes with the deterministic actor (SUPE: deterministic high level, held for k steps);
    returns mean success / return / length."""
    rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
    stateful = hasattr(agent, 'init_eval_state')
    stats = defaultdict(list)
    for _ in range(num_episodes):
        observation, info = env.reset()
        goal = info['goal']
        agent_state = agent.init_eval_state() if stateful else None
        done, step, ep_return = False, 0, 0.0
        while not done:
            rng, key = jax.random.split(rng)
            if stateful:
                action, agent_state = agent.sample_actions_with_state(
                    observations=observation, goals=goal, agent_state=agent_state, seed=key, temperature=temperature
                )
            else:
                action = agent.sample_actions(observation, goal, seed=key, temperature=temperature)
            action = np.clip(np.array(action), -1, 1)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            ep_return += float(reward)
        stats['success'].append(float(info.get('success', 0.0)))
        stats['episode_return'].append(ep_return)
        stats['episode_length'].append(step)
    return {k: float(np.mean(v)) for k, v in stats.items()}
