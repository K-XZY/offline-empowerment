"""CF-FlowBC-Empowerment baseline: action empowerment I(s+; a | s) with an InfoNCE critic and flow-matching negatives.

Per (s, a, s+) with s+ a discounted future state, N negative actions a_i ~ pi_bc(. | s) are drawn at the same s and
    b = log(N + 1) + log_softmax([T(s+, a, s), T(s+, a_1, s), ..., T(s+, a_N, s)])[0].

  pi_bc(a | s):     flow-matching BC policy (negatives, and query-time joints)
  f_dyn(s+ | s, a): flow-matching model of the discounted future (query-time joints)
  T(s+, a, s):      mlp critic, only trained after `pretrain_steps` (pi_bc / f_dyn keep training after that)

E(s) at arbitrary states is estimated from M joints a ~ pi_bc, s+ ~ f_dyn scored against fresh pi_bc negatives.
"""

from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


class VelocityField(nn.Module):
    """Rectified-flow velocity field v(x_t, t, cond)."""

    hidden_dims: Sequence[int]
    out_dim: int
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x_t, t, cond):
        h = jnp.concatenate([x_t, t, cond], axis=-1)
        return MLP((*self.hidden_dims, self.out_dim), activate_final=False, layer_norm=self.layer_norm)(h)


class CriticT(nn.Module):
    """Scalar critic T(s+, a, s)."""

    hidden_dims: Sequence[int]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, s_plus, a, s):
        h = jnp.concatenate([s_plus, a, s], axis=-1)
        return MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)(h).squeeze(-1)


class EmpowermentCRLFlowBCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _flow_loss(self, module_name, x, cond, rng, grad_params):
        """Conditional flow matching: regress v(x_t, t, cond) onto x - x_0."""
        k0, k1 = jax.random.split(rng)
        x0 = jax.random.normal(k0, x.shape)
        t = jax.random.uniform(k1, (*x.shape[:-1], 1))
        x_t = (1 - t) * x0 + t * x
        v = self.network.select(module_name)(x_t, t, cond, params=grad_params)
        return jnp.mean(jnp.square(v - (x - x0)))

    def _flow_sample(self, module_name, cond, rng, out_dim, clip=None, params=None):
        """Euler-integrate the flow from x_0 ~ N(0, I)."""
        x = jax.random.normal(rng, (*cond.shape[:-1], out_dim))
        n = self.config['flow_steps']
        dt = 1.0 / n

        def step(i, x):
            t = jnp.full((*cond.shape[:-1], 1), i * dt)
            return x + dt * self.network.select(module_name)(x, t, cond, params=params)

        x = jax.lax.fori_loop(0, n, step, x)
        if clip is not None:
            x = jnp.clip(x, -clip, clip)
        return x

    def sample_bc(self, observations, rng):
        return self._flow_sample('bc_flow', observations, rng, self.config['action_dim'], clip=1.0)

    def sample_dyn(self, observations, actions, rng):
        cond = jnp.concatenate([observations, actions], axis=-1)
        return self._flow_sample('dyn_flow', cond, rng, self.config['obs_dim'])

    def _b_values(self, s, a, sp, rng, grad_params):
        """Per-element InfoNCE bound with N fresh pi_bc negatives at each element's own s (no grad into pi_bc)."""
        B = s.shape[0]
        N = self.config['num_negatives']
        s_rep = jnp.broadcast_to(s[:, None, :], (B, N, s.shape[-1]))
        neg = jax.lax.stop_gradient(self._flow_sample('bc_flow', s_rep, rng, self.config['action_dim'], clip=1.0))
        pos_logit = self.network.select('critic')(sp, a, s, params=grad_params)  # [B]
        neg_logits = self.network.select('critic')(
            jnp.broadcast_to(sp[:, None, :], (B, N, sp.shape[-1])), neg, s_rep, params=grad_params
        )  # [B, N]
        logits = jnp.concatenate([pos_logit[:, None], neg_logits], axis=1)
        return jnp.log(N + 1.0) + jax.nn.log_softmax(logits, axis=-1)[:, 0]

    def _pretrain_losses(self, batch, grad_params, rng):
        k_bc, k_dyn, k_neg = jax.random.split(rng, 3)
        s, a, sp = batch['observations'], batch['actions'], batch['value_goals']
        bc_loss = self._flow_loss('bc_flow', a, s, k_bc, grad_params)
        dyn_loss = self._flow_loss('dyn_flow', sp, jnp.concatenate([s, a], axis=-1), k_dyn, grad_params)
        in_critic = self.network.step >= self.config['pretrain_steps']
        return bc_loss, dyn_loss, in_critic, k_neg

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        bc_loss, dyn_loss, in_critic, k_neg = self._pretrain_losses(batch, grad_params, rng)
        s, a, sp = batch['observations'], batch['actions'], batch['value_goals']
        N = self.config['num_negatives']

        # skip the (expensive) negative sampling entirely during pretraining
        def critic_branch(_):
            return jnp.log(N + 1.0) - jnp.mean(self._b_values(s, a, sp, k_neg, grad_params))

        critic_loss = jax.lax.cond(in_critic, critic_branch, lambda _: jnp.float32(0.0), None)
        info = {
            'bc_loss': bc_loss,
            'dyn_loss': dyn_loss,
            'critic_loss': critic_loss,
            'mi_estimate': jnp.where(in_critic, jnp.log(N + 1.0) - critic_loss, 0.0),
            'mi_cap': jnp.log(N + 1.0),
            'phase/in_critic': in_critic.astype(jnp.float32),
        }
        return bc_loss + dyn_loss + critic_loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=lambda p: self.total_loss(batch, p, rng=rng))
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def empowerment(self, observations, rng=None):
        """E(s) from M joints (a ~ pi_bc, s+ ~ f_dyn), each scored against N pi_bc negatives. [B, D] -> [B]."""
        rng = rng if rng is not None else self.rng
        B = observations.shape[0]
        M = self.config['est_num_joints']
        k_a, k_sp, k_neg = jax.random.split(rng, 3)

        s = jnp.repeat(observations, M, axis=0)
        a = self.sample_bc(s, k_a)
        sp = self.sample_dyn(s, a, k_sp)
        return self._b_values(s, a, sp, k_neg, None).reshape(B, M).mean(axis=1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        config = dict(config)
        config['action_dim'] = ex_actions.shape[-1]
        config['obs_dim'] = ex_observations.shape[-1]

        flow_dims = tuple(config['flow_hidden_dims'])
        bc_def = VelocityField(flow_dims, config['action_dim'], config['layer_norm'])
        dyn_def = VelocityField(flow_dims, config['obs_dim'], config['layer_norm'])
        critic_def = CriticT(tuple(config['critic_hidden_dims']), config['layer_norm'])

        ex_t = jnp.zeros((*ex_observations.shape[:-1], 1))
        ex_sa = jnp.concatenate([ex_observations, ex_actions], axis=-1)
        network_def = ModuleDict(dict(bc_flow=bc_def, dyn_flow=dyn_def, critic=critic_def))
        network_params = network_def.init(
            init_rng,
            bc_flow=(ex_actions, ex_t, ex_observations),
            dyn_flow=(ex_observations, ex_t, ex_sa),
            critic=(ex_observations, ex_actions, ex_observations),
        )['params']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='empowerment_crl_flowbc',
            lr=3e-4,
            batch_size=256,
            flow_hidden_dims=(512, 512, 512),
            critic_hidden_dims=(512, 512, 512),
            layer_norm=True,
            flow_steps=8,  # euler steps for flow sampling
            num_negatives=7,  # N, the bound is capped at log(N + 1)
            pretrain_steps=100000,  # pi_bc / f_dyn-only steps before the critic trains
            est_num_joints=16,  # M joints per state in empowerment()
            discount=0.99,
            # s+ is a geometrically-sampled future state of the same trajectory
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
        )
    )
