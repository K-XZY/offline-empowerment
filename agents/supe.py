"""SUPE (Wilcoxson et al. 2025): an online high level over frozen OPAL skills, with optimistic offline pseudo-labels.

Port of the online stage of github.com/rail-berkeley/supe (train_finetuning_supe.py, supe/agents/{rm,rnd}.py,
supe/agents/sac/sac_learner.py, supe/data/chunk_dataset.py, supe/wrappers/meta_env_wrapper.py).

  offline (once): agents/opal.py via train_offline.py -- the OPAL VAE q(z | s_{1:H}, a_{1:H}), p(z | s_1),
      pi(a | s, z). Loaded frozen here (`skill_checkpoint_path`).
  online: one replay row per macro step (s_t, u_t, R_t, mask_t):
      * high level: SAC over u = tanh(z) in (-1, 1)^D, tanh-gaussian actor, `num_qs` layer-normed critics with
        `num_min_qs` random heads min'ed in the target, no entropy backup, learned temperature (init 0.05, target -D/2).
      * low level: the frozen decoder runs z = arctanh(u) for `skill_commitment_k` (= H) env steps, sampling
        a ~ pi(a | s, z).
      * macro reward R = sum_i gamma_low^i (r_i + reward_shift); reward_shift=-1 turns our 0/1 rewards into SUPE's -1/0.
      * RLPD with the offline data: every window [t, t + H) inside one trajectory is labeled u_t = tanh(mean of
        q(z | window)), and at every update offline rows get reward = the dataset minimum ('min') or r_hat(s, u)
        ('pred'), and mask = sigmoid(m_hat(s, u)) from a reward model fit on the online rows.
      * RND: r += rnd_coeff * mean_j (f_hat(s, u) - f(s, u))_j^2 on online / offline rows, predictor fit online.
      * warm-up: the first `warmup_env_steps` env steps draw skills from the OPAL prior p(z | s).
      * one `update` = `critic_updates_per_update` critic minibatch steps (+ polyak) then one actor + temperature
        step on the last minibatch, called `utd_ratio` times per macro row.
      * aux_schedule='paper': the reward model (`update_rm`) and RND predictor (`update_rnd`, newest transition)
        are trained once per macro step as in SUPE, instead of inside `update` ('fused').
      * minibatch_split='interleave': every critic minibatch is exactly half online / half offline (SUPE `combine`).

  goal_conditioned=True: actor, critic and reward model take concat(s, g), g the episode's goal observation; offline
  rows get a random task goal from a bank of the env's task goals. RND, the prior and E' stay goal-free.

  Empowerment bonus (add_explore='distill' | 'distill-to-rlpd', as in agents/online_crl.py): a twin-head E'(s, u)
  regressed onto each row's trajectory-max empowerment, and the actor loss becomes
  alpha log pi - (mean_q Q + bonus_scale(t) * w * E'(s, u)).
"""

import copy
import json
import os
from typing import Any, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from agents.online_crl import DISTILL_MODES, _latest_epoch, load_empowerment_estimator, skill_empowerment_fast
from agents.opal import OPALAgent
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent
from utils.networks import GCValue, LogParam, TransformedWithMode

OFFLINE_RELABEL_TYPES = ('min', 'pred')
TANH_EPS = 1e-5  # SUPE `TanhConverter`


def to_tanh(latent):
    """Latent skill z -> replay/actor space u = tanh(z), clipped away from +-1 (SUPE `TanhConverter.to_tanh`)."""
    return jnp.clip(jnp.tanh(latent), -1.0 + TANH_EPS, 1.0 - TANH_EPS)


def from_tanh(u):
    """Replay/actor space u -> the decoder's latent z = arctanh(u) (SUPE `TanhConverter.from_tanh`)."""
    return jnp.arctanh(jnp.clip(u, -1.0 + TANH_EPS, 1.0 - TANH_EPS))


def load_opal_skills(seed, ex_observations, ex_actions, config):
    """Rebuild and restore the frozen continuous OPAL run named by `skill_checkpoint_path`."""
    ckpt_path = config['skill_checkpoint_path']
    assert ckpt_path is not None, 'supe needs --agent.skill_checkpoint_path=<opal run dir>'
    ckpt_path = ckpt_path.rstrip('/')
    with open(os.path.join(ckpt_path, 'flags.json')) as f:
        skill_flags = json.load(f)
    skill_config = skill_flags['agent']
    assert skill_config['agent_name'] == 'opal' and skill_config.get('latent_type', 'continuous') == 'continuous', (
        f'{ckpt_path} is not a continuous opal run'
    )
    restore_epoch = config['skill_restore_epoch']
    if restore_epoch is None:
        restore_epoch = _latest_epoch(ckpt_path)
    skill_agent = OPALAgent.create(seed, ex_observations, ex_actions, skill_config)
    skill_agent = restore_agent(skill_agent, ckpt_path, int(restore_epoch))
    resolved = dict(
        ckpt_path=ckpt_path,
        restore_epoch=int(restore_epoch),
        agent_name='opal',
        env_name=skill_flags.get('env_name'),
        skill_config=skill_config,
    )
    return skill_agent, resolved


# ── Networks (SUPE's `MLP` / `TanhNormal` / `StateActionValue` / `StateActionFeature`) ──────────────


class SupeMLP(nn.Module):
    """SUPE's `supe/networks/mlp.py` MLP: xavier-uniform Dense, optional LayerNorm BEFORE the ReLU."""

    hidden_dims: Sequence[int]
    activate_final: bool = False
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=nn.initializers.xavier_uniform())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = nn.relu(x)
        return x


class TanhGaussianActor(nn.Module):
    """SUPE's `TanhNormal`: state-dependent diagonal Gaussian in R^D squashed by tanh into (-1, 1)^D."""

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations, temperature=1.0):
        x = SupeMLP(self.hidden_dims, activate_final=True)(observations)
        means = nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        log_stds = nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        base = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        return TransformedWithMode(base, distrax.Block(distrax.Tanh(), ndims=1))


class EnsembleQ(nn.Module):
    """`num_qs` independent Q(s, u) MLPs (SUPE `Ensemble(StateActionValue)`), output [num_qs, B]."""

    hidden_dims: Sequence[int]
    num_qs: int
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, actions):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        q_cls = nn.vmap(
            SupeMLP,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        out = q_cls((*self.hidden_dims, 1), activate_final=False, use_layer_norm=self.use_layer_norm)(inputs)
        return out.squeeze(-1)


class StateActionMLP(nn.Module):
    """MLP over concat(s, u) -> `output_dim` (the reward-model heads and the RND feature nets)."""

    hidden_dims: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, observations, actions):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        x = SupeMLP(self.hidden_dims, activate_final=True)(inputs)
        out = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        if self.output_dim == 1:
            out = out.squeeze(-1)
        return out


# Parameter groups, each with its own Adam (SUPE keeps one TrainState per network).
PARAM_GROUPS = dict(
    actor=('modules_actor',),
    critic=('modules_critic',),
    alpha=('modules_alpha',),
    rm=('modules_rm_reward', 'modules_rm_mask'),
    rnd=('modules_rnd_predictor',),
    distill=('modules_distill_critic',),
)


class SUPEAgent(flax.struct.PyTreeNode):
    """SUPE high-level agent: SAC over OPAL latent skills with RLPD, learned reward/mask relabelling and RND."""

    rng: Any
    network: Any  # TrainState holding every module's params (no optimiser: see `opt_states`)
    opt_states: Any  # {group: optax state} for the groups in PARAM_GROUPS that exist
    skill_agent: Any  # frozen continuous OPAL run (saved with the agent, never trained)
    config: Any = nonpytree_field()
    emp_agent: Any = None  # frozen empowerment estimator (distilled bonus only)
    emp_mean: Any = None  # E_mean, set by `with_empowerment_mean` before the first update

    @property
    def uses_empowerment(self):
        return self.emp_agent is not None

    @property
    def uses_distill_bonus(self):
        return self.config['add_explore'] in DISTILL_MODES

    # ── Tanh-space helpers ────────────────────────────────────────────────────

    def example_skill(self):
        """The stored high-level action is u in (-1, 1)^D."""
        return np.zeros((int(self.config['skill_dim']),), dtype=np.float32)

    def _single_obs(self, observations):
        return observations.ndim == 1

    @property
    def stores_task_goals(self):
        """Replay rows carry `task_goals` (the episode's goal observation)."""
        return bool(self.config['goal_conditioned'])

    def _hi(self, observations, goals):
        """High-level network input: concat(s, g) when goal-conditioned, else s."""
        if not self.config['goal_conditioned']:
            return observations
        return jnp.concatenate([observations, goals], axis=-1)

    def _lr(self, group):
        return float(
            {'rm': self.config['rm_lr'], 'rnd': self.config['rnd_lr'], 'distill': self.config['distill_lr']}.get(
                group, self.config['lr']
            )
        )

    def _tx(self, group):
        return optax.adam(learning_rate=self._lr(group))

    # ── Empowerment estimator (distilled bonus) -- same API as agents/online_crl.py ────────

    @jax.jit
    def empowerment(self, observations, seed):
        return skill_empowerment_fast(self.emp_agent, observations, seed)

    def empowerment_np(self, observations, seed, chunk_size=1024):
        observations = np.asarray(observations)
        out = np.empty((len(observations),), dtype=np.float32)
        for start in range(0, len(observations), chunk_size):
            seed, key = jax.random.split(seed)
            chunk = observations[start : start + chunk_size]
            out[start : start + len(chunk)] = np.asarray(self.empowerment(jnp.asarray(chunk), key))
        return out

    def with_empowerment_mean(self, mean):
        return self.replace(emp_mean=jnp.asarray(mean, dtype=jnp.float32))

    def bonus_scale_at(self, env_steps):
        """Actor weight on E'(s, u): constant `bonus_scale`, or linearly decayed to 0 by
        `explore_reward_time_frac * total_env_steps` env steps (agents/online_crl.py `bonus_scale_at`)."""
        scale0 = float(self.config['bonus_scale'])
        frac = self.config['explore_reward_time_frac']
        total = self.config['total_env_steps']
        if frac is None or total is None:
            return jnp.asarray(scale0, dtype=jnp.float32)
        frac = float(frac)
        if frac <= 0.0:
            return jnp.asarray(0.0, dtype=jnp.float32)
        progress = jnp.asarray(env_steps, dtype=jnp.float32) / (frac * float(total))
        return scale0 * jnp.clip(1.0 - progress, 0.0, 1.0)

    # ── Relabelling (SUPE's per-batch offline reward/mask + RND bonus) ────────

    def rnd_reward(self, params, observations, actions):
        """rnd_coeff * mean_j (f_hat - f)_j^2 per row (SUPE `RND.get_reward`), at `params`."""
        pred = self.network.select('rnd_predictor')(observations, actions, params=params)
        target = self.network.select('rnd_target')(observations, actions, params=params)
        return self.config['rnd_coeff'] * jnp.mean(jnp.square(pred - target), axis=-1)

    def relabel(self, params, batch):
        """Rewards and masks the critic trains on, plus the task-only rewards the reward model trains on.

        Offline rows (`is_offline`): reward <- `offline_relabel` ('min': the constant
        `offline_min_reward`, 'pred': r_hat(s, u)), mask <- sigmoid(m_hat(s, u)). Then the RND bonus is
        added on the rows `use_rnd_online` / `use_rnd_offline` select. Returns
        (rewards_with_bonus, masks, task_rewards, rnd_bonus).
        """
        obs, u = batch['observations'], batch['actions']
        obs_g = self._hi(obs, batch.get('task_goals'))
        is_offline = batch['is_offline']
        if self.config['offline_relabel'] == 'min':
            offline_reward = jnp.full_like(batch['rewards'], float(self.config['offline_min_reward']))
        else:
            offline_reward = self.network.select('rm_reward')(obs_g, u, params=params)
        offline_mask = jax.nn.sigmoid(self.network.select('rm_mask')(obs_g, u, params=params))
        rewards = jnp.where(is_offline > 0, offline_reward, batch['rewards'])
        masks = jnp.where(is_offline > 0, offline_mask, batch['masks'])
        bonus = self.rnd_reward(params, obs, u)  # goal-free novelty
        use = (
            float(self.config['use_rnd_online']) * (1.0 - is_offline)
            + float(self.config['use_rnd_offline']) * is_offline
        )
        return rewards + use * bonus, masks, rewards, bonus

    # ── Losses (each reads `params`, differentiates w.r.t. its own group's subtree) ────────

    def critic_loss(self, batch, params, rng):
        """SUPE `update_critic`: r + gamma * mask * min over `num_min_qs` random heads of Q_targ(s', u')."""
        sample_rng, subset_rng = jax.random.split(rng)
        rewards, masks, task_rewards, bonus = self.relabel(params, batch)
        goals = batch.get('task_goals')
        next_obs_g = self._hi(batch['next_observations'], goals)  # the goal is fixed within an episode
        next_dist = self.network.select('actor')(next_obs_g, params=params)
        next_u, next_log_prob = next_dist.sample_and_log_prob(seed=sample_rng)
        next_qs = self.network.select('target_critic')(next_obs_g, next_u, params=params)  # [num_qs, B]
        num_qs, num_min_qs = int(self.config['num_qs']), int(self.config['num_min_qs'])
        if num_min_qs < num_qs:
            idx = jax.random.choice(subset_rng, num_qs, shape=(num_min_qs,), replace=False)
            next_qs = next_qs[idx]
        next_q = jnp.min(next_qs, axis=0)
        target_q = rewards + self.config['discount'] * masks * next_q
        if self.config['backup_entropy']:
            alpha = self.network.select('alpha')(params=params)
            target_q = target_q - self.config['discount'] * masks * alpha * next_log_prob
        target_q = jax.lax.stop_gradient(target_q)

        qs = self.network.select('critic')(
            self._hi(batch['observations'], goals), batch['actions'], params=params
        )  # [num_qs, B]
        critic_loss = jnp.mean(jnp.square(qs - target_q[None]))
        online = 1.0 - batch['is_offline']
        num_online = jnp.maximum(online.sum(), 1.0)
        num_offline = jnp.maximum(batch['is_offline'].sum(), 1.0)
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
            'target_q_mean': target_q.mean(),
            'reward_mean': rewards.mean(),
            'task_reward_online_mean': (task_rewards * online).sum() / num_online,
            'rnd_bonus_online_mean': (bonus * online).sum() / num_online,
            'rnd_bonus_offline_mean': (bonus * batch['is_offline']).sum() / num_offline,
            'mask_offline_mean': (masks * batch['is_offline']).sum() / num_offline,
            'mask_online_mean': (masks * online).sum() / num_online,
            'frac_offline': batch['is_offline'].mean(),
        }

    def rm_loss(self, batch, params, active):
        """Reward model (SUPE `RM._update`): MSE reward head + BCE termination head on the ONLINE rows, RND-free."""
        obs_g, u = self._hi(batch['observations'], batch.get('task_goals')), batch['actions']
        weight = (1.0 - batch['is_offline']) * active
        num_rows = jnp.maximum(weight.sum(), 1.0)
        r_hat = self.network.select('rm_reward')(obs_g, u, params=params)
        m_logit = self.network.select('rm_mask')(obs_g, u, params=params)
        r_loss = (jnp.square(r_hat - batch['rewards']) * weight).sum() / num_rows
        m_loss = (optax.sigmoid_binary_cross_entropy(m_logit, batch['masks']) * weight).sum() / num_rows
        # Validation on the offline rows against their STORED (dataset) mask -- SUPE `RM.evaluate`.
        off = batch['is_offline']
        num_off = jnp.maximum(off.sum(), 1.0)
        val_m_loss = (optax.sigmoid_binary_cross_entropy(m_logit, batch['masks']) * off).sum() / num_off
        return r_loss + m_loss, {
            'r_loss': r_loss,
            'm_loss': m_loss,
            'val_m_loss_offline': val_m_loss,
            'r_hat_online_mean': (r_hat * weight).sum() / num_rows,
            'mask_hat_online_mean': (jax.nn.sigmoid(m_logit) * weight).sum() / num_rows,
            'active': active,
        }

    def rnd_loss(self, batch, params, active):
        """RND predictor (SUPE `RND.update`): MSE to the frozen target on the ONLINE (s, u) rows."""
        obs, u = batch['observations'], batch['actions']
        weight = (1.0 - batch['is_offline']) * active
        num_rows = jnp.maximum(weight.sum(), 1.0)
        pred = self.network.select('rnd_predictor')(obs, u, params=params)
        target = self.network.select('rnd_target')(obs, u, params=params)
        per_row = jnp.mean(jnp.square(pred - target), axis=-1)
        loss = (per_row * weight).sum() / num_rows
        return loss, {'predictor_loss': loss, 'active': active}

    def distill_loss(self, batch, params):
        """Twin-head E'(s, u) regressed onto `distill_target` - E_mean (agents/online_crl.py `distill_loss`)."""
        weight = batch['distill_target_ready']
        if self.config['add_explore'] == 'distill':
            weight = weight * (1.0 - batch['is_offline'])
        target = jnp.where(weight > 0, batch['distill_target'] - self.emp_mean, 0.0)
        es = self.network.select('distill_critic')(batch['observations'], actions=batch['actions'], params=params)
        sq_err = jnp.square(es - target[None])
        num_rows = jnp.maximum(weight.sum(), 1.0)
        loss = (sq_err * weight[None]).sum() / (num_rows * es.shape[0])
        target_mean = (target * weight).sum() / num_rows
        target_var = (jnp.square(target - target_mean) * weight).sum() / num_rows
        online = 1.0 - batch['is_offline']
        return loss, {
            'distill_loss': loss,
            'explained_variance': 1.0 - loss / jnp.maximum(target_var, 1e-8),
            'e_mean': (es.mean(axis=0) * weight).sum() / num_rows,
            'target_mean': target_mean,
            'target_std': jnp.sqrt(target_var),
            'frac_rows_fit': weight.mean(),
            'frac_online_rows_ready': (batch['distill_target_ready'] * online).sum() / jnp.maximum(online.sum(), 1.0),
        }

    def bonus_value(self, params, batch, actions):
        """(twin-min E'(s, u) [B], row weight [B]) -- the distilled bonus at the actor's actions."""
        values = self.network.select('distill_critic')(batch['observations'], actions=actions, params=params)
        value = jnp.min(values, axis=0)
        if self.config['add_explore'] == 'distill':
            weight = 1.0 - batch['is_offline']
        else:
            weight = jnp.ones_like(value)
        return value, weight

    def actor_loss(self, batch, params, rng, env_steps):
        """SUPE `update_actor`: alpha log pi - mean_over_heads Q(s, u ~ pi), plus the optional distilled bonus."""
        obs_g = self._hi(batch['observations'], batch.get('task_goals'))
        dist = self.network.select('actor')(obs_g, params=params)
        u, log_probs = dist.sample_and_log_prob(seed=rng)
        qs = self.network.select('critic')(obs_g, u, params=params)
        q = qs.mean(axis=0)
        alpha = self.network.select('alpha')(params=params)
        info = {'entropy': -log_probs.mean(), 'q_pi_mean': q.mean(), 'u_abs_mean': jnp.abs(u).mean()}
        q_total = q
        if self.uses_distill_bonus:
            bonus, weight = self.bonus_value(params, batch, u)
            scale = self.bonus_scale_at(env_steps)
            q_total = q + scale * weight * bonus
            info['bonus_mean'] = (bonus * weight).sum() / jnp.maximum(weight.sum(), 1.0)
            info['bonus_scale'] = scale
        actor_loss = (alpha * log_probs - q_total).mean()
        info['actor_loss'] = actor_loss
        return actor_loss, info

    def alpha_loss(self, entropy, params):
        """SUPE `update_temperature`: alpha * (H - H_target), H the batch entropy (stop-gradient)."""
        alpha = self.network.select('alpha')(params=params)
        loss = alpha * (jax.lax.stop_gradient(entropy) - float(self.config['target_entropy']))
        return loss, {'alpha': alpha, 'alpha_loss': loss}

    # ── One Adam step of one parameter group ──────────────────────────────────

    def _group_step(self, params, opt_states, group, loss_fn):
        """Differentiate `loss_fn(full_params)` w.r.t. `group`'s subtree only and apply its Adam step."""
        keys = PARAM_GROUPS[group]
        sub = {key: params[key] for key in keys}

        def wrapped(sub_params):
            return loss_fn({**params, **sub_params})

        (_, info), grads = jax.value_and_grad(wrapped, has_aux=True)(sub)
        updates, new_opt_state = self._tx(group).update(grads, opt_states[group], sub)
        new_sub = optax.apply_updates(sub, updates)
        return {**params, **new_sub}, {**opt_states, group: new_opt_state}, info

    def _polyak(self, params):
        tau = float(self.config['tau'])
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1.0 - tau), params['modules_critic'], params['modules_target_critic']
        )
        return {**params, 'modules_target_critic': new_target}

    # ── Update ────────────────────────────────────────────────────────────────

    @jax.jit
    def update(self, batch, env_steps=None):
        """One SUPE update: `critic_updates_per_update` critic (+ reward model, + E') minibatch steps, then
        `rnd_updates_per_update` RND steps and one actor + alpha step. `env_steps` gates the reward-model / RND
        training starts and drives the distilled bonus annealing."""
        num_steps = int(self.config['critic_updates_per_update'])
        batch_size = batch['observations'].shape[0]
        assert batch_size % num_steps == 0, (
            f'batch_size ({batch_size}) must be a multiple of critic_updates_per_update ({num_steps})'
        )
        mini = batch_size // num_steps
        new_rng, perm_rng, scan_rng, actor_rng = jax.random.split(self.rng, 4)
        env_steps = jnp.asarray(1e18 if env_steps is None else env_steps, dtype=jnp.float32)
        rm_active = (env_steps >= float(self.config['rm_start_env_steps'])).astype(jnp.float32)
        rnd_active = (env_steps >= float(self.config['rnd_start_env_steps'])).astype(jnp.float32)

        if self.config['minibatch_split'] == 'interleave':
            # SUPE `combine(offline, online)` then a contiguous reshape: minibatch j holds the j-th slice of the
            # online rows and the j-th slice of the offline rows, so with a 50/50 batch every critic step sees
            # exactly mini/2 of each. (Stable sort: online rows first, each part keeps the sampler's random order.)
            order = jnp.argsort(batch['is_offline'], stable=True)
            assert mini % 2 == 0, f'minibatch_split=interleave needs an even minibatch, got {mini}'
            half = num_steps * (mini // 2)
            order = jnp.concatenate(
                [order[:half].reshape(num_steps, mini // 2), order[half:].reshape(num_steps, mini // 2)], axis=1
            ).reshape(-1)
        else:
            # Random minibatches so every critic step sees online and offline rows.
            order = jax.random.permutation(perm_rng, batch_size)
        minibatches = jax.tree_util.tree_map(lambda x: x[order].reshape((num_steps, mini) + x.shape[1:]), batch)
        fused_aux = self.config['aux_schedule'] == 'fused'

        def scan_step(carry, mb):
            params, opt_states, rng = carry
            rng, critic_rng = jax.random.split(rng)
            params, opt_states, critic_info = self._group_step(
                params, opt_states, 'critic', lambda p: self.critic_loss(mb, p, critic_rng)
            )
            params = self._polyak(params)
            info = {f'critic/{k}': v for k, v in critic_info.items()}
            if fused_aux:
                params, opt_states, rm_info = self._group_step(
                    params, opt_states, 'rm', lambda p: self.rm_loss(mb, p, rm_active)
                )
                info.update({f'rm/{k}': v for k, v in rm_info.items()})
            if self.uses_distill_bonus:
                params, opt_states, distill_info = self._group_step(
                    params, opt_states, 'distill', lambda p: self.distill_loss(mb, p)
                )
                info.update({f'distill/{k}': v for k, v in distill_info.items()})
            return (params, opt_states, rng), info

        (params, opt_states, _), scan_info = jax.lax.scan(
            scan_step, (self.network.params, self.opt_states, scan_rng), minibatches
        )
        info = jax.tree_util.tree_map(lambda x: x.mean(axis=0), scan_info)

        # RND predictor on the online rows of the last minibatch(es) ('fused' only; 'paper' -> update_rnd).
        rnd_info = {}
        for i in range(int(self.config['rnd_updates_per_update']) if fused_aux else 0):
            mb = jax.tree_util.tree_map(lambda x: x[num_steps - 1 - (i % num_steps)], minibatches)
            params, opt_states, rnd_info = self._group_step(
                params, opt_states, 'rnd', lambda p: self.rnd_loss(mb, p, rnd_active)
            )
        info.update({f'rnd/{k}': v for k, v in rnd_info.items()})

        # Actor + temperature on the last minibatch (SUPE `update`).
        last = jax.tree_util.tree_map(lambda x: x[-1], minibatches)
        params, opt_states, actor_info = self._group_step(
            params, opt_states, 'actor', lambda p: self.actor_loss(last, p, actor_rng, env_steps)
        )
        params, opt_states, alpha_info = self._group_step(
            params, opt_states, 'alpha', lambda p: self.alpha_loss(actor_info['entropy'], p)
        )
        info.update({f'actor/{k}': v for k, v in actor_info.items()})
        info.update({f'alpha/{k}': v for k, v in alpha_info.items()})
        info['schedule/env_steps'] = env_steps

        network = self.network.replace(params=params)
        return self.replace(network=network, opt_states=opt_states, rng=new_rng), info

    @jax.jit
    def update_rm(self, batch):
        """aux_schedule='paper': SUPE `rm.update(online_batch, utd_ratio)` -- `rm_updates_per_macro` sequential
        steps on contiguous slices of an ONLINE batch of rm_updates_per_macro x rm_batch_size rows. Called once
        per macro step by train_online.py from `rm_start_env_steps` on."""
        num_steps = int(self.config['rm_updates_per_macro'])
        size = batch['observations'].shape[0]
        assert size % num_steps == 0, f'rm batch ({size}) must be a multiple of rm_updates_per_macro ({num_steps})'
        slices = jax.tree_util.tree_map(lambda x: x.reshape((num_steps, size // num_steps) + x.shape[1:]), batch)
        one = jnp.ones((), dtype=jnp.float32)

        def step(carry, mb):
            params, opt_states = carry
            params, opt_states, info = self._group_step(params, opt_states, 'rm', lambda p: self.rm_loss(mb, p, one))
            return (params, opt_states), info

        (params, opt_states), info = jax.lax.scan(step, (self.network.params, self.opt_states), slices)
        info = jax.tree_util.tree_map(lambda x: x[-1], info)
        return self.replace(network=self.network.replace(params=params), opt_states=opt_states), {
            f'rm/{k}': v for k, v in info.items()
        }

    @jax.jit
    def update_rnd(self, transition):
        """aux_schedule='paper': SUPE `rnd.update` -- ONE predictor step on the single newest online (s, u).
        `transition` holds unbatched `observations` / `actions`. Called once per macro step from
        `rnd_start_env_steps` on."""
        mb = dict(
            observations=jnp.asarray(transition['observations'], dtype=jnp.float32)[None],
            actions=jnp.asarray(transition['actions'], dtype=jnp.float32)[None],
            is_offline=jnp.zeros((1,), dtype=jnp.float32),
        )
        one = jnp.ones((), dtype=jnp.float32)
        params, opt_states, info = self._group_step(
            self.network.params, self.opt_states, 'rnd', lambda p: self.rnd_loss(mb, p, one)
        )
        return self.replace(network=self.network.replace(params=params), opt_states=opt_states), {
            f'rnd/{k}': v for k, v in info.items()
        }

    # ── Acting: high level ────────────────────────────────────────────────────

    @jax.jit
    def sample_skills(self, observations, goals=None, seed=None, temperature=1.0, env_steps=None):
        """u ~ pi_hi(. | s[, g]) in tanh space (temperature=0 -> tanh(mean)). `goals` is used only with
        `goal_conditioned` (plain SUPE is task-conditioned through the reward). With `env_steps` <
        `warmup_env_steps` the skill is drawn from the OPAL prior p(z | s) instead (SUPE's `start_training`
        warm-up; goal-free, as in the paper)."""
        if seed is None:
            seed = self.rng
        single = self._single_obs(observations)
        obs_b = observations[None, ...] if single else observations
        goals_b = None
        if self.config['goal_conditioned']:
            assert goals is not None, 'supe: goal_conditioned=True needs the goal at acting time'
            goals_b = goals[None, ...] if single else goals
        prior_seed, actor_seed = jax.random.split(seed)
        u = self.network.select('actor')(self._hi(obs_b, goals_b), temperature=temperature).sample(seed=actor_seed)
        if env_steps is not None:
            prior_u = to_tanh(self.skill_agent.network.select('prior')(obs_b, 1.0).sample(seed=prior_seed))
            warm = jnp.asarray(env_steps, dtype=jnp.float32) < float(self.config['warmup_env_steps'])
            u = jnp.where(warm, prior_u, u)
        return u[0] if single else u

    # ── Acting: low level (frozen OPAL decoder) ───────────────────────────────

    @jax.jit
    def low_level_actions(self, observations, skills, seed=None):
        """a ~ pi(a | s, z = arctanh(u)) from the frozen OPAL decoder."""
        if seed is None:
            seed = self.rng
        z = from_tanh(jnp.asarray(skills, dtype=jnp.float32))
        return self.skill_agent.sample_actions_with_skill(
            observations, z, seed=seed, temperature=float(self.config['low_temperature'])
        )

    # ── Evaluation (stateful: hold u for k steps) ─────────────────────────────

    def init_eval_state(self):
        # Keyed `z` (not `skill`): the evaluator's skill-usage histogram is for integer skills.
        return {'z': jnp.zeros((int(self.config['skill_dim']),), dtype=jnp.float32), 'count': jnp.zeros((), jnp.int32)}

    @jax.jit
    def sample_actions_with_state(self, observations, goals=None, agent_state=None, seed=None, temperature=1.0):
        """Pick u every `skill_commitment_k` steps (temperature=0 -> the actor's tanh(mean)), hold it, decode."""
        if seed is None:
            seed = self.rng
        if agent_state is None:
            agent_state = self.init_eval_state()
        skill_seed, action_seed = jax.random.split(seed)
        k = int(self.config['skill_commitment_k'])
        reselect = (agent_state['count'] % k) == 0
        fresh = self.sample_skills(observations, goals, seed=skill_seed, temperature=temperature)
        u = jnp.where(reselect, fresh, agent_state['z'])
        action = self.low_level_actions(observations, u, seed=action_seed)
        return action, {'z': u, 'count': agent_state['count'] + 1}

    # ── Offline window labelling (RLPD; utils/rlpd.py) ────────────────────────

    @jax.jit
    def label_chunk_skills(self, observations_seq, actions_seq, seq_mask, seed):
        """u = tanh(mean of q(z | window)) per window: float32 [B, D] (SUPE `ChunkDataset`, `VAE.encode`)."""
        del seq_mask, seed  # deterministic posterior mean; partial windows are dropped by the keep mask
        skill_dim = int(self.config['skill_dim'])
        enc_out = self.skill_agent.network.select('encoder')(observations_seq, actions_seq)
        return to_tanh(enc_out[..., :skill_dim])

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of LOW-LEVEL env actions (shapes the frozen OPAL decoder).
            config: Configuration dictionary (`get_config`).
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, skill_rng = jax.random.split(rng, 3)
        config = copy.deepcopy(config)

        if config['offline_relabel'] not in OFFLINE_RELABEL_TYPES:
            raise ValueError(
                f'offline_relabel must be one of {OFFLINE_RELABEL_TYPES}, got {config["offline_relabel"]!r}'
            )
        add_explore = config['add_explore']
        if add_explore is not None and add_explore not in DISTILL_MODES:
            raise ValueError(f'supe: add_explore must be None or one of {DISTILL_MODES}, got {add_explore!r}')
        if add_explore is not None and config['emp_checkpoint_path'] is None:
            raise ValueError(
                f'supe: add_explore={add_explore!r} needs --agent.emp_checkpoint_path (the frozen estimator).'
            )

        # Frozen continuous OPAL skills.
        skill_agent, resolved = load_opal_skills(seed, ex_observations, ex_actions, config)
        skill_config = resolved['skill_config']
        skill_dim = int(skill_config['skill_dim'])
        chunk_size = int(skill_config['chunk_size'])
        k = config['skill_commitment_k']
        k = chunk_size if k is None else int(k)
        if k != chunk_size:
            print(f'[supe] WARNING: skill_commitment_k={k} != OPAL chunk_size={chunk_size} (SUPE uses k == H)')
        if config['gamma_low'] is None:
            config['gamma_low'] = float(config['discount'])  # NB SUPE sums a chunk with OPAL's discount (0.99)
        if config['target_entropy'] is None:
            config['target_entropy'] = -skill_dim / 2.0  # SUPE `SACLearner.create`
        if config['rm_start_env_steps'] is None:
            config['rm_start_env_steps'] = 2 * int(config['warmup_env_steps'])  # SUPE `start_training_rm`
        if config['rnd_start_env_steps'] is None:
            config['rnd_start_env_steps'] = 2 * int(config['warmup_env_steps'])
        if config['min_replay_size'] is None:
            config['min_replay_size'] = -(-int(config['warmup_env_steps']) // k)  # ceil: rows of the warm-up
        if config['utd_ratio'] is None:
            config['utd_ratio'] = float(k)  # one update() call per env step, the online_crl budget
        # The dataset's minimum macro reward (SUPE `ds_minr`): our offline data has no rewards, so every full
        # chunk's reward is reward_shift summed with the within-chunk discount.
        gamma_low = float(config['gamma_low'])
        config['offline_min_reward'] = float(config['reward_shift']) * float(sum(gamma_low**i for i in range(k)))
        if config['aux_schedule'] not in ('fused', 'paper'):
            raise ValueError(f"aux_schedule must be 'fused' or 'paper', got {config['aux_schedule']!r}")
        if config['minibatch_split'] not in ('random', 'interleave'):
            raise ValueError(f"minibatch_split must be 'random' or 'interleave', got {config['minibatch_split']!r}")
        if config['minibatch_split'] == 'interleave' and float(config['offline_ratio']) != 0.5:
            raise ValueError('minibatch_split=interleave assumes 50/50 RLPD batches (offline_ratio=0.5, SUPE combine).')
        if int(config['batch_size']) % int(config['critic_updates_per_update']) != 0:
            raise ValueError(
                f'batch_size={config["batch_size"]} must be a multiple of critic_updates_per_update='
                f'{config["critic_updates_per_update"]} (one minibatch per critic step).'
            )
        assert 1 <= int(config['num_min_qs']) <= int(config['num_qs']), 'need 1 <= num_min_qs <= num_qs'

        # Empowerment estimator for the distilled bonus (optional).
        emp_agent, emp_mean_metric = None, None
        if add_explore is not None:
            emp_agent, config['emp_restore_epoch'], emp_mean_metric = load_empowerment_estimator(
                seed, ex_observations, ex_actions, config
            )

        obs_dim = ex_observations.shape[-1]
        ex_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
        # Goal-conditioned: the goal is a full observation (`info['goal']`), so concat(s, g) is 2 x obs_dim.
        ex_obs_g = jnp.zeros((1, 2 * obs_dim if config['goal_conditioned'] else obs_dim), dtype=jnp.float32)
        ex_u = jnp.zeros((1, skill_dim), dtype=jnp.float32)
        hidden = tuple(config['hidden_dims'])
        critic_def = EnsembleQ(hidden, num_qs=int(config['num_qs']), use_layer_norm=bool(config['critic_layer_norm']))
        network_info = dict(
            actor=(TanhGaussianActor(hidden, skill_dim), (ex_obs_g,)),
            critic=(critic_def, (ex_obs_g, ex_u)),
            target_critic=(copy.deepcopy(critic_def), (ex_obs_g, ex_u)),
            alpha=(LogParam(init_value=float(config['init_temperature'])), ()),
            rm_reward=(StateActionMLP(tuple(config['rm_hidden_dims']), 1), (ex_obs_g, ex_u)),
            rm_mask=(StateActionMLP(tuple(config['rm_hidden_dims']), 1), (ex_obs_g, ex_u)),
            rnd_predictor=(
                StateActionMLP(tuple(config['rnd_hidden_dims']), int(config['rnd_feature_dim'])),
                (ex_obs, ex_u),
            ),
            rnd_target=(
                StateActionMLP(tuple(config['rnd_hidden_dims']), int(config['rnd_feature_dim'])),
                (ex_obs, ex_u),
            ),
        )
        if add_explore is not None:
            # Same E'(s, u) network as online_crl's distill_critic (twin GELU MLP, LayerNorm).
            distill_def = GCValue(hidden_dims=tuple(config['distill_hidden_dims']), layer_norm=True, ensemble=True)
            network_info['distill_critic'] = (distill_def, (ex_obs, None, ex_u))
        networks = {key: value[0] for key, value in network_info.items()}
        network_args = {key: value[1] for key, value in network_info.items()}
        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params['modules_target_critic'] = network_params['modules_critic']
        network = TrainState.create(network_def, network_params, tx=None)

        opt_states = {}
        for group, keys in PARAM_GROUPS.items():
            if all(key in network_params for key in keys):
                lr = {'rm': config['rm_lr'], 'rnd': config['rnd_lr'], 'distill': config['distill_lr']}.get(
                    group, config['lr']
                )
                opt_states[group] = optax.adam(learning_rate=float(lr)).init({key: network_params[key] for key in keys})

        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        stored_config.update(
            skill_dim=skill_dim,
            skill_commitment_k=k,
            skill_checkpoint_path=resolved['ckpt_path'],
            skill_restore_epoch=resolved['restore_epoch'],
            skill_agent_name=resolved['agent_name'],
            skill_env_name=resolved['env_name'],
            skill_chunk_size=chunk_size,
            num_skills=None,  # continuous latent: no finite skill set
            emp_mean_metric=emp_mean_metric,
        )

        print(
            f'[supe] OPAL skills D={skill_dim}, k={k}; {config["num_qs"]} Qs (min over {config["num_min_qs"]}), '
            f'discount={config["discount"]}, gamma_low={config["gamma_low"]}, utd_ratio={config["utd_ratio"]}, '
            f'aux_schedule={config["aux_schedule"]}, minibatch_split={config["minibatch_split"]}, '
            f'goal_conditioned={config["goal_conditioned"]}, offline reward={config["offline_relabel"]} '
            f'({config["offline_min_reward"]:.4f}), add_explore={add_explore}'
        )
        return cls(
            rng,
            network=network,
            opt_states=opt_states,
            skill_agent=skill_agent,
            config=flax.core.FrozenDict(**stored_config),
            emp_agent=emp_agent,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='supe',
            # ── Frozen OPAL skills (agents/opal.py, latent_type=continuous) ──
            skill_checkpoint_path=ml_collections.config_dict.placeholder(str),  # run dir (flags.json + params_*.pkl)
            skill_restore_epoch=ml_collections.config_dict.placeholder(int),  # None -> latest
            skill_commitment_k=ml_collections.config_dict.placeholder(
                int
            ),  # None -> the checkpoint's chunk_size (SUPE)
            low_temperature=1.0,  # decoder sampling temperature (SUPE samples a ~ pi(a|s,z) in train and eval)
            # ── SMDP reward (SUPE MetaPolicyActionWrapper) ──
            reward_shift=-1.0,  # added to every env reward: 0/1 -> -1/0 (SUPE's SparseRewardWrapper convention)
            gamma_low=ml_collections.config_dict.placeholder(float),  # within-chunk discount; None -> discount
            discount=0.99,  # high-level discount per macro step (SUPE: 0.99; 0.995 for cube / antsoccer / humanoid)
            # ── SAC high level (SUPE configs/rlpd_config.py + sac_config.py, OGBench widths) ──
            lr=3e-4,  # actor / critic / temperature learning rate
            hidden_dims=(512, 512, 512),  # actor + critic MLP widths (SUPE's `is_ogbench` override)
            num_qs=10,  # critic ensemble size (RLPD)
            num_min_qs=2,  # random heads min'ed in the target (SUPE: 1 on antmaze/antsoccer/humanoid, 2 on cube/scene)
            critic_layer_norm=True,
            tau=0.005,  # Polyak rate of the target critic
            init_temperature=0.05,
            target_entropy=ml_collections.config_dict.placeholder(float),  # None -> -skill_dim / 2
            backup_entropy=False,  # SUPE passes --config.backup_entropy=False everywhere
            # ── Update schedule ──
            # Default = the online CRL budget (one 1024-row step per env step). SUPE's own schedule is
            # batch_size=5120, critic_updates_per_update=20, utd_ratio=4.
            batch_size=1024,  # rows per update() call (= minibatch rows x critic_updates_per_update)
            critic_updates_per_update=1,  # critic minibatch steps per update() call
            rnd_updates_per_update=1,  # RND predictor minibatch steps per update() call (aux_schedule='fused' only)
            aux_schedule='fused',  # 'fused': RM + RND inside update() | 'paper': update_rm / update_rnd per macro step
            rm_updates_per_macro=20,  # aux_schedule='paper': RM minibatch steps per macro step (SUPE utd_ratio)
            rm_batch_size=128,  # aux_schedule='paper': online rows per RM step (SUPE batch_size * (1 - offline_ratio))
            minibatch_split='random',  # 'random' permutation | 'interleave' (exactly 50/50 per critic minibatch)
            unroll_length=1,  # update every macro row ...
            utd_ratio=ml_collections.config_dict.placeholder(float),  # ... this many times; None -> k (1 per env step)
            min_replay_size=ml_collections.config_dict.placeholder(
                int
            ),  # rows before the 1st update; None -> warm-up / k
            replay_size=250000,  # macro rows (SUPE keeps the whole run: 1M env steps / k=4)
            warmup_env_steps=5000,  # prior-skill warm-up (SUPE start_training)
            rm_start_env_steps=ml_collections.config_dict.placeholder(int),  # reward model starts; None -> 2 x warm-up
            rnd_start_env_steps=ml_collections.config_dict.placeholder(
                int
            ),  # RND predictor starts; None -> 2 x warm-up
            # ── Goal conditioning (multigoal envs) ──
            goal_conditioned=False,  # actor / critic / reward model on concat(s, g); offline rows get random task goals
            goal_bank_resets_per_task=16,  # resets per task when collecting goal observations for offline rows
            # ── RLPD / offline pseudo-labels ──
            offline_ratio=0.5,  # share of every batch drawn from the offline buffer (SUPE offline_ratio)
            offline_relabel='min',  # 'min' (SUPE state-based default) | 'pred' (kitchen)
            offline_full_windows_only=True,  # only windows inside one trajectory are anchors (SUPE ChunkDataset)
            use_rnd_online=True,
            use_rnd_offline=True,
            label_chunk_bytes=64 * 1024 * 1024,  # offline windows pushed through the encoder at a time
            # ── Reward model + RND (SUPE configs/rm_config.py, rnd_config.py) ──
            rm_hidden_dims=(256, 256, 256),
            rm_lr=3e-4,
            rnd_hidden_dims=(256, 256, 256),
            rnd_feature_dim=256,
            rnd_coeff=8.0,
            rnd_lr=3e-4,
            # ── Distilled empowerment bonus (optional; same keys as agents/online_crl.py) ──
            add_explore=ml_collections.config_dict.placeholder(str),  # None | 'distill' | 'distill-to-rlpd'
            emp_checkpoint_path=ml_collections.config_dict.placeholder(str),  # frozen estimator run dir
            emp_restore_epoch=ml_collections.config_dict.placeholder(int),
            emp_num_splus_samples=64,
            emp_mean=ml_collections.config_dict.placeholder(float),  # override E_mean (else checkpoint metric / rows)
            bonus_scale=1.0,  # actor weight on E'(s, u)
            explore_reward_time_frac=ml_collections.config_dict.placeholder(
                float
            ),  # anneal bonus_scale to 0 by this fraction of total_env_steps
            distill_hidden_dims=(512, 512, 512),
            distill_lr=3e-4,
            total_env_steps=ml_collections.config_dict.placeholder(int),  # set by train_online.py
        )
    )
    return config
