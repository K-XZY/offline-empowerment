"""Offline empowerment estimator (ours, Sec. 4 of the paper).

Jointly learns the discounted occupancies Q^z(s+ | s, a), V^z(s+ | s) of a skill-conditioned policy pi(a | s, z) by
TD on offline transitions, and trains pi to maximize the effective empowerment computed from them.
log Q / log V are -||phi - psi(s+)||^2 / d, i.e. the occupancy of psi(s+) is N(phi, d/2 I) in latent space.
"""

from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from jax.scipy.special import logsumexp

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor


def log1mexp(x):
    """Stable log(1 - exp(x)) for x <= 0."""
    log_half = -0.6931471805599453
    return jnp.where(x < log_half, jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))


def log_diff_exp(log_total, log_part):
    """Stable log(exp(log_total) - exp(log_part)), needs log_part <= log_total."""
    return log_total + log1mexp(log_part - log_total)


def clipped_linexp_loss(target, pred, gamma, t=5.0):
    """Clipped linexp loss in log space; regresses e^-pred to gamma * e^-target (paper eq. 18)."""
    target, t = jax.lax.stop_gradient(target), jax.lax.stop_gradient(t)

    p0 = target + t
    value_p0 = gamma * jnp.exp(t) - (target + t)
    slope_p0 = gamma * jnp.exp(t) - 1.0

    true_loss = gamma * jnp.exp(pred - target) - pred
    linear_loss = value_p0 + slope_p0 * (pred - p0)

    loss = jnp.where(pred - target < t, true_loss, linear_loss)
    return loss.mean()


class EmpowermentQNetwork(nn.Module):
    """log Q^z(s+ | s, a) = -||phi(s, a, z) - psi(s+)||^2 / d."""

    hidden_dims: Sequence[int]
    latent_dim: int = 128
    layer_norm: bool = True
    shared_psi: Optional[nn.Module] = None

    def setup(self):
        self.phi_net = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def phi(self, observations, actions, skills):
        return self.phi_net(jnp.concatenate([observations, actions, skills], axis=-1))

    def psi(self, future_states):
        return self.shared_psi(future_states)

    def __call__(self, observations, actions, skills, future_states):
        diff = self.phi(observations, actions, skills) - self.psi(future_states)
        # log p(psi | phi) for psi ~ N(phi, (d/2) I), up to a constant
        return -jnp.sum(diff**2, axis=-1) / self.latent_dim


class EmpowermentVNetwork(nn.Module):
    """log V^z(s+ | s) = -||phi(s, z) - psi(s+)||^2 / d."""

    hidden_dims: Sequence[int]
    latent_dim: int = 128
    layer_norm: bool = True
    shared_psi: Optional[nn.Module] = None

    def setup(self):
        self.phi_net = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def phi(self, observations, skills):
        return self.phi_net(jnp.concatenate([observations, skills], axis=-1))

    def psi(self, future_states):
        return self.shared_psi(future_states)

    def __call__(self, observations, skills, future_states):
        diff = self.phi(observations, skills) - self.psi(future_states)
        return -jnp.sum(diff**2, axis=-1) / self.latent_dim


class SharedPsiEncoder(nn.Module):
    """psi(s+), shared by Q and V."""

    hidden_dims: Sequence[int]
    latent_dim: int = 128
    layer_norm: bool = True

    def setup(self):
        self.psi_net = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, future_states):
        return self.psi_net(future_states)


class SkillConditionedActor(GCActor):
    """Actor conditioned on a one-hot skill instead of a goal (unit std, only the mean is learned)."""

    def __call__(self, observations, skills):
        means = self.mean_net(self.actor_net(jnp.concatenate([observations, skills], axis=-1)))
        return distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.ones_like(means))


class EmpowermentAgent(flax.struct.PyTreeNode):
    """Offline empowerment agent. Networks: q, v, target_q, target_v, policy."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _sample_skills(self, rng, batch_size):
        K = self.config['num_skills']
        skills = jax.random.randint(rng, (batch_size,), 0, K)
        return skills, jnp.eye(K)[skills]

    def _policy_actions(self, observations, skills_onehot, params, rng, num_noise_samples=None):
        """a = clip(mu(s, z) + eta, -1, 1), eta ~ N(0, sigma_a^2 I) with a fixed sigma_a (`action_noise_std`).

        `num_noise_samples=M` draws M noise samples around the same mean -> [M, batch, A].
        """
        actions = self.network.select('policy')(observations, skills_onehot, params=params).mode()
        shape = actions.shape if num_noise_samples is None else (num_noise_samples, *actions.shape)
        noise = jax.random.normal(rng, shape) * self.config['action_noise_std']
        return jnp.clip(actions + noise, -1.0, 1.0)

    @staticmethod
    def _logits_from_embeddings(phi_emb, psi_emb, latent_dim):
        """log p(psi | phi) for psi ~ N(phi, (d/2) I), up to a constant."""
        return -jnp.sum((phi_emb - psi_emb) ** 2, axis=-1) / latent_dim

    def _v_phi_all_skills(self, observations, use_target=False):
        """phi_V(s, z) for all K skills -> [K, batch, d] (no grad)."""
        K = self.config['num_skills']
        batch_size = observations.shape[0]
        key = 'target_v' if use_target else 'v'
        net = self.network.model_def.modules[key]
        vars_ = jax.lax.stop_gradient({'params': self.network.params[f'modules_{key}']})

        def phi_for_skill(z_onehot):
            z_batch = jnp.repeat(z_onehot[None, :], batch_size, axis=0)
            return net.apply(vars_, observations, z_batch, method=net.phi)

        return jax.vmap(phi_for_skill)(jnp.eye(K))

    def _q_phi(self, observations, actions, skills_onehot, use_target=False):
        """phi_Q(s, a, z) with frozen params; the gradient only enters through `actions`."""
        key = 'target_q' if use_target else 'q'
        net = self.network.model_def.modules[key]
        vars_ = jax.lax.stop_gradient({'params': self.network.params[f'modules_{key}']})
        return net.apply(vars_, observations, actions, skills_onehot, method=net.phi)

    def empowerment(self, observations, rng, sample_chunk_size=None):
        """Monte Carlo estimate of E(s) = I(s+; z | s) (paper eq. 15) -> [batch], in nats.

        Skills are looped with lax.scan and the s+ samples with a chunked lax.map to bound peak memory.
        """
        K = self.config['num_skills']
        num_samples = self.config['num_splus_samples']
        d = self.config['value_latent_dim']
        log_K = jnp.log(K)
        if sample_chunk_size is None:
            sample_chunk_size = self.config.get('emp_sample_chunk_size', 64)
        sample_chunk_size = max(1, min(int(sample_chunk_size), num_samples))

        _, sample_rng = jax.random.split(rng)
        skill_rngs = jax.random.split(sample_rng, K)

        phi_all = self._v_phi_all_skills(observations)  # [K, batch, d]

        def empowerment_for_skill(carry, xs):
            phi_z, skill_rng = xs
            noise = jax.random.normal(skill_rng, (num_samples, *phi_z.shape))
            psi_samples = phi_z[None] + noise * jnp.sqrt(d / 2.0)

            def contribution(psi_splus):
                log_v = self._logits_from_embeddings(phi_z, psi_splus, d)  # [batch]
                log_v_all = self._logits_from_embeddings(phi_all, psi_splus, d)  # [K, batch]
                return log_v - (logsumexp(log_v_all, axis=0) - log_K)

            contributions = jax.lax.map(contribution, psi_samples, batch_size=sample_chunk_size)  # [N, batch]
            return carry, contributions.mean(axis=0)

        _, emp_per_skill = jax.lax.scan(empowerment_for_skill, None, (phi_all, skill_rngs))  # [K, batch]
        return emp_per_skill.mean(axis=0)

    def q_loss(self, batch, grad_params, skills_onehot, rng):
        """Q(s+ | s, a) -> gamma V(s+ | s') plus the self term Q(s | s, a) -> (1 - gamma) + gamma V(s | s').

        Dataset actions get the same N(0, sigma_a^2) noise as the policy's actions.
        """
        discount = self.config['discount']
        future = batch['value_goals']
        noise = jax.random.normal(jax.random.fold_in(rng, 7), batch['actions'].shape) * self.config['action_noise_std']
        actions = jnp.clip(batch['actions'] + noise, -1.0, 1.0)

        log_q = self.network.select('q')(batch['observations'], actions, skills_onehot, future, params=grad_params)
        log_v_next_future = self.network.select('target_v')(batch['next_observations'], skills_onehot, future)
        loss_future = clipped_linexp_loss(target=-log_v_next_future, pred=-log_q, gamma=discount)

        # self term, with s+ = s
        log_q_current = self.network.select('q')(
            batch['observations'], actions, skills_onehot, batch['observations'], params=grad_params
        )
        log_v_next_current = self.network.select('target_v')(
            batch['next_observations'], skills_onehot, batch['observations']
        )
        log_current_target = jnp.logaddexp(jnp.log(1.0 - discount), jnp.log(discount) + log_v_next_current)
        loss_current = clipped_linexp_loss(target=-log_current_target, pred=-log_q_current, gamma=1.0)

        loss = loss_future + loss_current
        return loss, {
            'q_loss': loss,
            'q_loss_future': loss_future,
            'q_loss_current': loss_current,
            'q_log_mean': log_q.mean(),
            'v_next_future_log_mean': log_v_next_future.mean(),
            'q_log_current_mean': log_q_current.mean(),
            'v_next_current_log_mean': log_v_next_current.mean(),
        }

    def v_loss(self, batch, grad_params, skills_onehot, rng):
        """V(s+ | s, z) -> Q(s+ | s, a', z), a' ~ pi(. | s, z) (eq. 9)."""
        future = batch['value_goals']
        actions_pi = self._policy_actions(batch['observations'], skills_onehot, params=None, rng=rng)
        log_q_pi = self.network.select('target_q')(batch['observations'], actions_pi, skills_onehot, future)
        log_v = self.network.select('v')(batch['observations'], skills_onehot, future, params=grad_params)
        loss = clipped_linexp_loss(target=-log_q_pi, pred=-log_v, gamma=1.0)

        return loss, {
            'v_loss': loss,
            'v_log_mean': log_v.mean(),
            'q_pi_log_mean': log_q_pi.mean(),
            'v_max': jnp.exp(log_v).max(),
            'v_min': jnp.exp(log_v).min(),
        }

    def bc_loss(self, batch, grad_params, skills_onehot):
        """-log pi(a | s, z) for dataset actions."""
        dist = self.network.select('policy')(batch['observations'], skills_onehot, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])
        loss = -log_prob.mean()
        return loss, {
            'bc_loss': loss,
            'bc_log_prob_mean': log_prob.mean(),
            'bc_log_prob_max': log_prob.max(),
            'bc_log_prob_min': log_prob.min(),
        }

    def policy_loss(self, batch, grad_params, skills, skills_onehot, rng):
        """Maximize the (pi, z)-modified empowerment E_Delta (eqs. 11-13).

        For each of N sample slots we draw z' ~ Unif(K) and psi ~ V^{z'}(. | s), and the per-sample reward is
            r = (M_{z'} / V_{z'}) log(M_{z'} / m_bar),
        with M_{z'} = Q^z(psi | s, a) if z' = z else V^{z'}(psi | s), and m_bar = (Q^z + sum_{z'' != z} V^{z''}) / K.
        All V's are frozen, so the policy gradient only flows through Q^z(psi | s, pi(s, z) + eta).
        """
        batch_size = batch['observations'].shape[0]
        K = self.config['num_skills']
        N = self.config['num_splus_samples']
        d = self.config['value_latent_dim']
        log_K = jnp.log(K)

        rng, sample_rng = jax.random.split(rng)
        act_rng = jax.random.fold_in(rng, 4)

        phi_all = self._v_phi_all_skills(batch['observations'])  # [K, batch, d]

        rng_zp, sample_rng = jax.random.split(sample_rng)
        z_prime = jax.random.randint(rng_zp, (N, batch_size), 0, K)  # [N, batch]

        # psi ~ V^{z'}(. | s)
        batch_idx = jnp.arange(batch_size)[None, :]
        phi_z_prime_v = phi_all[z_prime, batch_idx]  # [N, batch, d]
        psi = phi_z_prime_v + jax.random.normal(sample_rng, (N, batch_size, d)) * jnp.sqrt(d / 2.0)

        # log Q^z(psi | s, pi(s, z) + eta), an independent eta per sample slot
        policy_actions = self._policy_actions(
            batch['observations'], skills_onehot, params=grad_params, rng=act_rng, num_noise_samples=N
        )  # [N, batch, A]
        phi_z_q = jax.vmap(lambda a: self._q_phi(batch['observations'], a, skills_onehot))(policy_actions)
        log_q = self._logits_from_embeddings(phi_z_q, psi, d)  # [N, batch]

        log_v_all = self._logits_from_embeddings(phi_all[None], psi[:, None], d)  # [N, K, batch]
        log_v_z = log_v_all[:, skills, jnp.arange(batch_size)]  # [N, batch]
        log_v_all_lse = logsumexp(log_v_all, axis=1)
        log_c_sg = jax.lax.stop_gradient(log_diff_exp(log_v_all_lse, log_v_z))  # log sum_{z'' != z} V^{z''}

        log_m_bar = logsumexp(jnp.stack([log_q, log_c_sg], axis=0), axis=0) - log_K

        log_v_z_prime_sg = jax.lax.stop_gradient(log_v_all[jnp.arange(N)[:, None], z_prime, batch_idx])
        log_m_z_prime = jnp.where(z_prime == skills[None, :], log_q, log_v_z_prime_sg)

        w = jnp.exp(log_m_z_prime - log_v_z_prime_sg)  # importance weight: Q^z / V^z if z' = z, else 1
        reward = w * (log_m_z_prime - log_m_bar)
        e_delta = reward.sum(axis=0)

        loss = -e_delta.mean()
        return loss, {
            'policy_loss': loss,
            'e_delta_mean': e_delta.mean(),
            'e_delta_max': e_delta.max(),
            'e_delta_min': e_delta.min(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        skills_rng, empowerment_rng = jax.random.split(rng, 2)
        q_act_rng, v_act_rng = jax.random.fold_in(rng, 101), jax.random.fold_in(rng, 102)
        policy_rng = jax.random.fold_in(rng, 103)
        skills, skills_onehot = self._sample_skills(skills_rng, batch['observations'].shape[0])
        info = {}

        bc_loss, bc_info = self.bc_loss(batch, grad_params, skills_onehot)
        info.update({f'bc/{k}': v for k, v in bc_info.items()})

        q_loss, q_info = self.q_loss(batch, grad_params, skills_onehot, q_act_rng)
        info.update({f'q/{k}': v for k, v in q_info.items()})

        v_loss, v_info = self.v_loss(batch, grad_params, skills_onehot, v_act_rng)
        info.update({f'v/{k}': v for k, v in v_info.items()})

        pi_loss, pi_info = self.policy_loss(batch, grad_params, skills, skills_onehot, policy_rng)
        info.update({f'policy/{k}': v for k, v in pi_info.items()})

        # E(s) is only logged, so only compute it on log steps (train update is at step % log_interval == 0,
        # the validation loss right after it at == 1)
        step_mod = self.network.step % self.config['log_interval']
        zero = jnp.zeros((), dtype=jnp.float32)

        def _emp_metrics():
            emp = jax.lax.stop_gradient(self.empowerment(batch['observations'], rng=empowerment_rng))
            return emp.mean(), emp.min(), emp.max()

        emp_mean, emp_min, emp_max = jax.lax.cond(
            (step_mod == 0) | (step_mod == 1), _emp_metrics, lambda: (zero, zero, zero)
        )
        info['empowerment/mean'] = emp_mean
        info['empowerment/min'] = emp_min
        info['empowerment/max'] = emp_max

        total = q_loss + v_loss + pi_loss + self.config['bc_alpha'] * bc_loss
        info['total_loss'] = total
        return total, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=lambda p: self.total_loss(batch, p, rng=rng))

        tau = self.config['tau']
        new_params = dict(new_network.params)
        for key in ('q', 'v'):
            new_params[f'modules_target_{key}'] = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1 - tau),
                new_network.params[f'modules_{key}'],
                new_network.params[f'modules_target_{key}'],
            )
        new_network = new_network.replace(params=new_params)
        return self.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]
        num_skills = config['num_skills']
        hidden_dims = tuple(config.get('value_hidden_dims', None) or config['hidden_dims'])
        latent_dim = config['value_latent_dim']
        layer_norm = config['layer_norm']

        policy_def = SkillConditionedActor(
            hidden_dims=tuple(config['actor_hidden_dims']),
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=True,
        )

        # Q and V have their own phi but share psi(s+); the target nets share their own target psi.
        def make_nets():
            shared_psi = SharedPsiEncoder(hidden_dims=hidden_dims, latent_dim=latent_dim, layer_norm=layer_norm)
            kwargs = dict(hidden_dims=hidden_dims, latent_dim=latent_dim, layer_norm=layer_norm, shared_psi=shared_psi)
            return EmpowermentQNetwork(**kwargs), EmpowermentVNetwork(**kwargs)

        q_def, v_def = make_nets()
        target_q_def, target_v_def = make_nets()

        ex_skills = jnp.eye(num_skills)[jnp.arange(ex_observations.shape[0]) % num_skills]
        network_def = ModuleDict(dict(q=q_def, v=v_def, target_q=target_q_def, target_v=target_v_def, policy=policy_def))
        network_params = network_def.init(
            init_rng,
            q=(ex_observations, ex_actions, ex_skills, ex_observations),
            v=(ex_observations, ex_skills, ex_observations),
            target_q=(ex_observations, ex_actions, ex_skills, ex_observations),
            target_v=(ex_observations, ex_skills, ex_observations),
            policy=(ex_observations, ex_skills),
        )['params']
        network_params['modules_target_q'] = network_params['modules_q']
        network_params['modules_target_v'] = network_params['modules_v']

        network = TrainState.create(network_def, network_params, tx=optax.adam(config['lr']))
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='empowerment_skill',
            lr=3e-4,
            batch_size=1024,
            hidden_dims=(512, 512, 512),  # Q / V hidden dims (unless value_hidden_dims is set)
            value_hidden_dims=ml_collections.config_dict.placeholder(tuple),
            value_latent_dim=256,  # d
            actor_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,  # target network update rate
            num_skills=15,  # K
            num_splus_samples=1,  # future-state samples per update
            bc_alpha=0.0,  # alpha_bc, behavioral constraint coefficient
            action_noise_std=0.1,  # sigma_a
            log_interval=5000,  # only compute the (logged) E(s) estimate on log steps; synced with --log_interval
            # s+ is a uniformly random dataset state
            value_p_trajgoal=0.0,
            value_p_randomgoal=1.0,
        )
    )
