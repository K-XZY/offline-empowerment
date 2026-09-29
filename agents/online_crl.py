"""Online contrastive RL (CRL), optionally with the empowerment bonus (ours) and / or an RND bonus.

CRL: bilinear critic phi(s, a)^T psi(g) / sqrt(d) (2 heads) trained with the in-batch binary NCE loss on future
states of the same trajectory, and a tanh-gaussian actor maximizing E[Q(s, a, g) - alpha log pi(a | s, g)] with an
auto-tuned alpha (target entropy -0.5 * dim(A)). The actor reads critic head 0.

Empowerment bonus (`add_explore=distill | distill-to-rlpd`, Sec. 5.2): a frozen offline estimator (an
`empowerment_skill` run, `emp_checkpoint_path`) gives E(s) for every replay row. A twin-head net B(s, a) is regressed
(MSE) onto max_k E(s_k) - E_mean over the row's whole trajectory (final observation included), and the actor
maximizes Q(s, a, g) + bonus_scale * min_i B_i(s, a). Online rows only get their target once their episode is over.
  'distill':         B is fit on, and added to the actor loss for, online rows only
  'distill-to-rlpd': B is fit on, and added for, online and offline (RLPD) rows
E_mean only shifts B by a constant: `emp_mean` if set, else the estimator's last logged training/empowerment/mean,
else the mean over the offline rows (or the warm-up online rows without RLPD).

RND bonus (`rnd_bonus_scale`, Burda et al. 2019): fixed random target f and predictor f_hat on the landing state s',
input whitened with running online-state stats and clipped to +-5; r = ||f_hat(s') - f(s')||^2 divided by the running
std of the discounted intrinsic return. It's learned into a twin critic Q_rnd(s, a) (polyak target, non-episodic
backup) and the actor adds rnd_bonus_scale * min_i Q_rnd_i(s, a). Predictor, stats, Q_rnd and the actor term all use
online rows only.

`explore_reward_time_frac` / `rnd_time_frac` (None -> constant) linearly anneal the two actor weights to 0 by that
fraction of `total_env_steps`.
"""

import copy
import csv
import glob
import json
import os
import re
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from jax.scipy.special import logsumexp

from agents.empowerment_skill import EmpowermentAgent
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent
from utils.networks import GCActor, GCBilinearValue, GCValue, LogParam, RNDEmbedding, RunningMeanStd

DISTILL_MODES = ('distill', 'distill-to-rlpd')


def _latest_epoch(run_dir):
    epochs = [int(re.search(r'params_(\d+)\.pkl$', p).group(1)) for p in glob.glob(os.path.join(run_dir, 'params_*.pkl'))]
    if not epochs:
        raise FileNotFoundError(f'no params_*.pkl in {run_dir}')
    return max(epochs)


def _read_last_metric(csv_path, column):
    """Last logged value of `column` in a train.csv (None if missing)."""
    if not os.path.exists(csv_path):
        return None
    value = None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get(column) not in (None, ''):
                value = float(row[column])
    return value


def load_empowerment_estimator(seed, ex_observations, ex_actions, config):
    """Rebuild the frozen estimator from its run dir (flags.json + params_<epoch>.pkl)."""
    ckpt_path = config['emp_checkpoint_path'].rstrip('/')
    with open(os.path.join(ckpt_path, 'flags.json')) as f:
        emp_config = dict(json.load(f)['agent'])
    assert emp_config['agent_name'] == 'empowerment_skill', 'emp_checkpoint_path must be an empowerment_skill run'
    emp_config['num_splus_samples'] = int(config['emp_num_splus_samples'])

    restore_epoch = config['emp_restore_epoch']
    if restore_epoch is None:
        restore_epoch = _latest_epoch(ckpt_path)
    estimator = EmpowermentAgent.create(seed, ex_observations, ex_actions, emp_config)
    estimator = restore_agent(estimator, ckpt_path, int(restore_epoch))
    mean_metric = _read_last_metric(os.path.join(ckpt_path, 'train.csv'), 'training/empowerment/mean')
    return estimator, int(restore_epoch), mean_metric


def skill_empowerment_fast(estimator, observations, rng):
    """Same as `EmpowermentAgent.empowerment` (same rng stream), with the squared distances to all K skill
    embeddings expanded into an einsum so the [N, K, B, d] tensor is never built."""
    K = int(estimator.config['num_skills'])
    N = int(estimator.config['num_splus_samples'])
    d = int(estimator.config['value_latent_dim'])
    log_K = jnp.log(K)

    _, sample_rng = jax.random.split(rng)
    skill_rngs = jax.random.split(sample_rng, K)

    phi_all = estimator._v_phi_all_skills(observations)  # [K, B, d]
    phi_sq = jnp.sum(phi_all**2, axis=-1)  # [K, B]

    def per_skill(carry, xs):
        phi_z, skill_rng = xs
        noise = jax.random.normal(skill_rng, (N, *phi_z.shape))
        psi = phi_z[None] + noise * jnp.sqrt(d / 2.0)  # [N, B, d]
        psi_sq = jnp.sum(psi**2, axis=-1)
        cross = jnp.einsum('kbd,nbd->knb', phi_all, psi)
        log_v_all = -(phi_sq[:, None, :] - 2.0 * cross + psi_sq[None]) / d  # [K, N, B]
        log_v = -jnp.sum((phi_z[None] - psi) ** 2, axis=-1) / d  # [N, B]
        log_denom = logsumexp(log_v_all, axis=0) - log_K
        return carry, (log_v - log_denom).mean(axis=0)

    _, emp_per_skill = jax.lax.scan(per_skill, None, (phi_all, skill_rngs))  # [K, B]
    return emp_per_skill.mean(axis=0)


class OnlineCRLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    emp_agent: Any = None  # frozen offline estimator (None -> no empowerment bonus)
    emp_mean: Any = None  # E_mean, set by `with_empowerment_mean` before the first update
    rnd_state: Any = None  # running RND stats (obs_rms, ret_rms, ret_carry)

    @property
    def uses_empowerment(self):
        return self.emp_agent is not None

    @property
    def uses_rnd(self):
        return self.config['rnd_bonus_scale'] is not None

    # ── Empowerment ───────────────────────────────────────────────────────────

    @jax.jit
    def empowerment(self, observations, seed):
        return skill_empowerment_fast(self.emp_agent, observations, seed)

    def empowerment_np(self, observations, seed, chunk_size=1024):
        """E(s) over a large array, in chunks (one key per chunk)."""
        observations = np.asarray(observations)
        out = np.empty((len(observations),), dtype=np.float32)
        for start in range(0, len(observations), chunk_size):
            seed, key = jax.random.split(seed)
            chunk = observations[start : start + chunk_size]
            out[start : start + len(chunk)] = np.asarray(self.empowerment(jnp.asarray(chunk), key))
        return out

    def with_empowerment_mean(self, mean):
        return self.replace(emp_mean=jnp.asarray(mean, dtype=jnp.float32))

    # ── Losses ────────────────────────────────────────────────────────────────

    def contrastive_loss(self, batch, grad_params):
        batch_size = batch['observations'].shape[0]

        v, phi, psi = self.network.select('critic')(
            batch['observations'], batch['value_goals'], actions=batch['actions'], info=True, params=grad_params
        )
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])  # [B, B, 2]
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        v = jnp.exp(v)
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': jnp.sum(logits * I) / jnp.sum(I),
            'logits_neg': jnp.sum(logits * (1 - I)) / jnp.sum(1 - I),
            'logits': logits.mean(),
        }

    def distill_loss(self, batch, grad_params):
        """MSE of B(s, a) (both heads) onto max_k E(s_k) - E_mean, on rows whose target is known."""
        weight = batch['distill_target_ready']
        if self.config['add_explore'] == 'distill':
            weight = weight * (1.0 - batch['is_offline'])
        target = jnp.where(weight > 0, batch['distill_target'] - self.emp_mean, 0.0)  # not-ready rows hold nan

        es = self.network.select('distill_critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        num_rows = jnp.maximum(weight.sum(), 1.0)
        loss = (jnp.square(es - target[None]) * weight[None]).sum() / (num_rows * es.shape[0])

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

    def rnd_normalize(self, observations):
        return self.rnd_state['obs_rms'].normalize(observations)

    def rnd_raw_reward(self, next_observations):
        x = self.rnd_normalize(next_observations)
        target = self.network.select('rnd_target')(x)
        pred = self.network.select('rnd_predictor')(x)
        return jnp.sum(jnp.square(pred - target), axis=-1)

    def rnd_reward(self, batch):
        ret_rms = self.rnd_state['ret_rms']
        return self.rnd_raw_reward(batch['next_observations']) / jnp.sqrt(ret_rms.var + ret_rms.eps)

    def rnd_predictor_loss(self, batch, grad_params, rng):
        """Predictor regression onto the fixed target, on a random `rnd_update_proportion` of the online rows."""
        x = self.rnd_normalize(batch['next_observations'])
        target = self.network.select('rnd_target')(x)
        pred = self.network.select('rnd_predictor')(x, params=grad_params)
        per_row = jnp.sum(jnp.square(pred - target), axis=-1)
        keep = jax.random.uniform(rng, per_row.shape) < self.config['rnd_update_proportion']
        weight = (1.0 - batch['is_offline']) * keep.astype(jnp.float32)
        loss = (per_row * weight).sum() / jnp.maximum(weight.sum(), 1.0)
        ret_rms = self.rnd_state['ret_rms']
        return loss, {
            'predictor_loss': loss,
            'ret_std': jnp.sqrt(ret_rms.var + ret_rms.eps),
            'frac_rows_fit': weight.mean(),
        }

    def rnd_critic_loss(self, batch, grad_params, rng):
        """Q_rnd(s, a) -> r + gamma * mask * min_i Q_rnd_targ_i(s', a'), a' ~ pi(. | s', g), online rows only."""
        reward = self.rnd_reward(batch)
        masks = jnp.ones_like(batch['masks']) if self.config['rnd_nonepisodic'] else batch['masks']
        weight = 1.0 - batch['is_offline']

        next_actions = self.network.select('actor')(batch['next_observations'], batch['actor_goals']).sample(seed=rng)
        next_q = jnp.min(self.network.select('target_rnd_critic')(batch['next_observations'], actions=next_actions), 0)
        target_q = reward + self.config['bonus_discount'] * masks * next_q

        qs = self.network.select('rnd_critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        loss = (jnp.square(qs - target_q[None]) * weight[None]).sum() / (jnp.maximum(weight.sum(), 1.0) * qs.shape[0])
        return loss, {
            'rnd_critic_loss': loss,
            'q_mean': qs.mean(),
            'reward_mean': reward.mean(),
            'reward_max': reward.max(),
            'target_q_mean': target_q.mean(),
        }

    @jax.jit
    def rnd_update_stats(self, observations, is_next, valid):
        """Update the RND running stats with newly visited states (in visit order).

        As in the paper, the intrinsic rewards of these states are computed with the old stats first; then the
        discounted intrinsic return (never reset at episode ends) is accumulated and its running std updated, and
        finally the observation stats absorb the states. `is_next` marks states that are some transition's s'
        (not an episode's reset state), `valid` masks padding.
        """
        state = self.rnd_state
        raw = self.rnd_raw_reward(observations)
        obs_rms = state['obs_rms'].update_masked(observations, valid)
        emit = jnp.asarray(valid, jnp.float32) * jnp.asarray(is_next, jnp.float32)
        gamma = self.config['bonus_discount']

        def filter_step(ret, xs):
            r, e = xs
            ret = jnp.where(e > 0, gamma * ret + r, ret)
            return ret, ret

        ret_carry, rets = jax.lax.scan(filter_step, state['ret_carry'], (raw, emit))
        ret_rms = state['ret_rms'].update_masked(rets, emit)
        info = {
            'int_rew_raw_mean': (raw * emit).sum() / jnp.maximum(emit.sum(), 1.0),
            'ret_std': jnp.sqrt(ret_rms.var + ret_rms.eps),
        }
        return self.replace(rnd_state=dict(obs_rms=obs_rms, ret_rms=ret_rms, ret_carry=ret_carry)), info

    def bonus_scale_at(self, env_steps, scale0, frac):
        """`scale0`, or linearly decayed to 0 by env step `frac * total_env_steps` if `frac` is set."""
        total = self.config['total_env_steps']
        if frac is None or total is None:
            return jnp.asarray(scale0, dtype=jnp.float32)
        if frac <= 0.0:
            return jnp.asarray(0.0, dtype=jnp.float32)
        progress = jnp.asarray(env_steps, dtype=jnp.float32) / (float(frac) * float(total))
        return float(scale0) * jnp.clip(1.0 - progress, 0.0, 1.0)

    def actor_loss(self, batch, grad_params, rng, env_steps):
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=rng)

        # critic / bonus nets at their stored params, the gradient only reaches the actor through the actions
        q = self.network.select('critic')(batch['observations'], batch['actor_goals'], actions=actions)[0]
        q_total = q
        info = {}

        if self.uses_empowerment:
            e = jnp.min(self.network.select('distill_critic')(batch['observations'], actions=actions), axis=0)
            if self.config['add_explore'] == 'distill':
                weight = 1.0 - batch['is_offline']
            else:
                weight = jnp.ones_like(e)
            scale = self.bonus_scale_at(env_steps, self.config['bonus_scale'], self.config['explore_reward_time_frac'])
            q_total = q_total + scale * weight * e
            info['bonus_scale_effective'] = scale
            info['bonus_term_mean'] = (scale * weight * e).mean()

        if self.uses_rnd:
            q_rnd = jnp.min(self.network.select('rnd_critic')(batch['observations'], actions=actions), axis=0)
            weight = 1.0 - batch['is_offline']
            scale = self.bonus_scale_at(env_steps, self.config['rnd_bonus_scale'], self.config['rnd_time_frac'])
            q_total = q_total + scale * weight * q_rnd
            info['rnd_bonus_scale_effective'] = scale
            info['rnd_bonus_term_mean'] = (scale * weight * q_rnd).mean()

        entropy = -jax.lax.stop_gradient(log_probs)
        alpha = self.network.select('alpha')()
        alpha_param = self.network.select('alpha')(params=grad_params)

        actor_loss = (alpha * log_probs - q_total).mean()
        alpha_loss = (alpha_param * (entropy - self.config['target_entropy'])).mean()
        total_loss = actor_loss + alpha_loss
        info.update(
            {
                'total_loss': total_loss,
                'actor_loss': actor_loss,
                'alpha_loss': alpha_loss,
                'alpha': alpha,
                'entropy': entropy.mean(),
                'q_pi_mean': q.mean(),
                'std': dist._distribution.stddev().mean(),
            }
        )
        return total_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, env_steps=None):
        info = {}
        rng = rng if rng is not None else self.rng
        actor_rng, rnd_rng, rnd_critic_rng = jax.random.split(rng, 3)

        loss, critic_info = self.contrastive_loss(batch, grad_params)
        info.update({f'critic/{k}': v for k, v in critic_info.items()})

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng, env_steps)
        info.update({f'actor/{k}': v for k, v in actor_info.items()})
        loss = loss + actor_loss

        # the losses below touch disjoint params, so one shared adam step == separate updates
        if self.uses_empowerment:
            distill_loss, distill_info = self.distill_loss(batch, grad_params)
            info.update({f'distill/{k}': v for k, v in distill_info.items()})
            loss = loss + distill_loss
        if self.uses_rnd:
            rnd_loss, rnd_info = self.rnd_predictor_loss(batch, grad_params, rnd_rng)
            info.update({f'rnd/{k}': v for k, v in rnd_info.items()})
            rnd_critic_loss, rnd_critic_info = self.rnd_critic_loss(batch, grad_params, rnd_critic_rng)
            info.update({f'rnd_critic/{k}': v for k, v in rnd_critic_info.items()})
            loss = loss + rnd_loss + rnd_critic_loss
        return loss, info

    @jax.jit
    def update(self, batch, env_steps=None):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng, env_steps=env_steps)
        )
        if self.uses_rnd:
            tau = self.config['bonus_tau']
            new_network.params['modules_target_rnd_critic'] = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1 - tau),
                self.network.params['modules_rnd_critic'],
                self.network.params['modules_target_rnd_critic'],
            )
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        return jnp.clip(dist.sample(seed=seed if seed is not None else self.rng), -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim
        if config['bonus_discount'] is None:
            config['bonus_discount'] = config['discount']

        critic_def = GCBilinearValue(
            hidden_dims=tuple(config['value_hidden_dims']),
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=False,
        )
        actor_def = GCActor(
            hidden_dims=tuple(config['actor_hidden_dims']),
            action_dim=action_dim,
            log_std_min=-5,
            tanh_squash=True,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            alpha=(LogParam(), ()),
        )

        def twin_critic():
            return GCValue(hidden_dims=tuple(config['value_hidden_dims']), layer_norm=config['layer_norm'], ensemble=True)

        emp_agent, emp_mean_metric = None, None
        if config['add_explore'] is not None:
            assert config['add_explore'] in DISTILL_MODES, f'add_explore must be one of {DISTILL_MODES}'
            assert config['emp_checkpoint_path'] is not None, 'the empowerment bonus needs --agent.emp_checkpoint_path'
            emp_agent, config['emp_restore_epoch'], emp_mean_metric = load_empowerment_estimator(
                seed, ex_observations, ex_actions, config
            )
            network_info['distill_critic'] = (twin_critic(), (ex_observations, None, ex_actions))

        uses_rnd = config['rnd_bonus_scale'] is not None
        if uses_rnd:
            rnd_critic_def = twin_critic()
            network_info['rnd_critic'] = (rnd_critic_def, (ex_observations, None, ex_actions))
            network_info['target_rnd_critic'] = (copy.deepcopy(rnd_critic_def), (ex_observations, None, ex_actions))
            rnd_kwargs = dict(hidden_dims=tuple(config['rnd_hidden_dims']), rep_dim=int(config['rnd_rep_dim']))
            network_info['rnd_target'] = (RNDEmbedding(**rnd_kwargs), (ex_observations,))
            network_info['rnd_predictor'] = (RNDEmbedding(**rnd_kwargs), (ex_observations,))

        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        if uses_rnd:
            # predictor has its own adam, the target never moves
            rnd_labels = {'modules_rnd_predictor': 'rnd_predictor', 'modules_rnd_target': 'rnd_target'}
            network_tx = optax.multi_transform(
                {
                    'agent': optax.adam(learning_rate=config['lr']),
                    'rnd_predictor': optax.adam(learning_rate=config['rnd_lr']),
                    'rnd_target': optax.set_to_zero(),
                },
                lambda params: {k: rnd_labels.get(k, 'agent') for k in params},
            )
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **{k: v[1] for k, v in network_info.items()})['params']
        if uses_rnd:
            network_params['modules_target_rnd_critic'] = network_params['modules_rnd_critic']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        rnd_state = None
        if uses_rnd:
            obs_shape = tuple(ex_observations.shape[1:])
            rnd_state = dict(
                obs_rms=RunningMeanStd(
                    mean=jnp.zeros(obs_shape, jnp.float32),
                    var=jnp.ones(obs_shape, jnp.float32),
                    count=jnp.float32(0.0),
                    clip_max=float(config['rnd_obs_clip']),
                ),
                ret_rms=RunningMeanStd(mean=jnp.float32(0.0), var=jnp.float32(1.0), count=jnp.float32(0.0)),
                ret_carry=jnp.float32(0.0),
            )

        config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        config['emp_mean_metric'] = emp_mean_metric
        return cls(rng, network=network, config=flax.core.FrozenDict(**config), emp_agent=emp_agent, rnd_state=rnd_state)


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='online_crl',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            latent_dim=512,
            layer_norm=True,
            discount=0.99,  # also the future-goal sampling discount
            target_entropy=ml_collections.config_dict.placeholder(float),  # None -> -mult * dim(A)
            target_entropy_multiplier=0.5,
            actor_fc_scale=0.01,
            # empowerment bonus
            add_explore=ml_collections.config_dict.placeholder(str),  # None | 'distill' | 'distill-to-rlpd'
            emp_checkpoint_path=ml_collections.config_dict.placeholder(str),  # empowerment_skill run dir
            emp_restore_epoch=ml_collections.config_dict.placeholder(int),  # None -> latest
            emp_num_splus_samples=64,  # future samples per skill for E(s)
            emp_mean=ml_collections.config_dict.placeholder(float),  # E_mean override
            bonus_scale=1.0,  # beta
            explore_reward_time_frac=ml_collections.config_dict.placeholder(float),  # None -> no annealing
            # RND bonus
            rnd_bonus_scale=ml_collections.config_dict.placeholder(float),  # beta_RND, None -> off
            rnd_time_frac=ml_collections.config_dict.placeholder(float),  # None -> no annealing
            rnd_hidden_dims=(512, 512),
            rnd_rep_dim=512,
            rnd_lr=1e-4,
            rnd_update_proportion=1.0,
            rnd_obs_clip=5.0,
            rnd_nonepisodic=True,
            bonus_discount=ml_collections.config_dict.placeholder(float),  # Q_rnd discount, None -> discount
            bonus_tau=0.005,  # Q_rnd target update rate
            total_env_steps=ml_collections.config_dict.placeholder(int),  # set by train_online.py
            # online schedule
            unroll_length=50,  # env steps between update rounds
            utd_ratio=1.0,  # updates per env step
            min_replay_size=1000,  # warm-up env steps
            replay_size=1000000,
            offline_ratio=0.5,  # RLPD: fraction of each batch from the offline data
        )
    )
