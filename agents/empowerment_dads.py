"""DADS-Empowerment baseline: skill empowerment I(s+; z | s) with a learned, MI-maximizing skill labeling.

On offline (s, s+) pairs (s+ a discounted future state), jointly learn
  alpha(z | s, s+ - s): soft skill assignment (only sees the endpoint pair),
  q(s+ - s | s, z):     mixture density channel,
  m(z | s):             skill marginal, fit to alpha by cross-entropy (stop-grad inside the bound),
to maximize the Barber-Agakov bound
  E_{z ~ alpha} [ log q(s+ | s, z) - log sum_z' m(z' | s) q(s+ | s, z') ].
`kl_coef` > 0 adds KL(Unif[k] || batch skill usage) to keep all skills in use.
"""

from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, default_init

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class MDN(nn.Module):
    """Diagonal-gaussian mixture density network -> (logits, means, log_stds)."""

    out_dim: int
    num_components: int = 8
    hidden_dims: Sequence[int] = (256, 256)
    layer_norm: bool = False

    @nn.compact
    def __call__(self, cond):
        h = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(cond)
        C, D = self.num_components, self.out_dim
        logits = nn.Dense(C, kernel_init=default_init())(h)
        means = nn.Dense(C * D, kernel_init=default_init(1e-2))(h).reshape(h.shape[:-1] + (C, D))
        log_stds = nn.Dense(C * D, kernel_init=default_init(1e-2))(h)
        log_stds = jnp.clip(log_stds.reshape(h.shape[:-1] + (C, D)), LOG_STD_MIN, LOG_STD_MAX)
        return logits, means, log_stds


def mdn_log_prob(logits, means, log_stds, x):
    log_w = jax.nn.log_softmax(logits, axis=-1)
    diff = (x[..., None, :] - means) / jnp.exp(log_stds)
    comp_lp = -0.5 * (diff**2 + 2.0 * log_stds + jnp.log(2.0 * jnp.pi)).sum(-1)
    return jax.nn.logsumexp(log_w + comp_lp, axis=-1)


def mdn_sample(key, logits, means, log_stds):
    k1, k2 = jax.random.split(key)
    comp = jax.random.categorical(k1, logits)
    mu = jnp.take_along_axis(means, comp[..., None, None], axis=-2)[..., 0, :]
    std = jnp.exp(jnp.take_along_axis(log_stds, comp[..., None, None], axis=-2)[..., 0, :])
    return mu + std * jax.random.normal(k2, mu.shape)


class EmpowermentDADSAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _q_ll_allz(self, s, x_delta, params=None):
        """log q(x | s, z) for every z -> [k, B]."""
        eye = jnp.eye(self.config['num_skills'])

        def per_z(zoh):
            cond = jnp.concatenate([s, jnp.tile(zoh, (s.shape[0], 1))], axis=-1)
            return mdn_log_prob(*self.network.select('q')(cond, params=params), x_delta)

        return jax.vmap(per_z)(eye)

    def _bracket_allz(self, s, x_delta, params=None):
        """b_z = log q(x | s, z) - logsumexp_z' (log m(z' | s) + log q(x | s, z')) for every z -> [k, B]."""
        lq_all = self._q_ll_allz(s, x_delta, params=params)
        logm = jax.lax.stop_gradient(jax.nn.log_softmax(self.network.select('clf')(s, params=params), axis=-1))
        log_mix = jax.nn.logsumexp(logm.T + lq_all, axis=0)
        return lq_all - log_mix[None]

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        s0 = batch['observations']
        target = batch['value_goals'] - s0

        logits = self.network.select('enc')(jnp.concatenate([s0, target], axis=-1), params=grad_params)
        alpha = jax.nn.softmax(logits, axis=-1)

        # expected bound under alpha, exact over z
        b_all = self._bracket_allz(s0, target, params=grad_params)
        bracket_loss = -(alpha * b_all.T).sum(-1).mean()

        # m(z | s) -> current assignment mix
        logm = jax.nn.log_softmax(self.network.select('clf')(s0, params=grad_params), axis=-1)
        clf_loss = -(jax.lax.stop_gradient(alpha) * logm).sum(-1).mean()

        # KL(Unif[k] || batch usage)
        k = self.config['num_skills']
        usage = alpha.mean(0)
        kl_usage = -jnp.log(k) - jnp.log(usage + 1e-8).mean()

        loss = bracket_loss + clf_loss + self.config['kl_coef'] * kl_usage
        usage = jax.lax.stop_gradient(usage)
        return loss, {
            'bracket_loss': bracket_loss,
            'clf_loss': clf_loss,
            'kl_usage': kl_usage,
            'mi_estimate': -jax.lax.stop_gradient(bracket_loss),
            'usage_min': usage.min(),
            'usage_max': usage.max(),
        }

    @jax.jit
    def update(self, batch):
        new_network, info = self.network.apply_loss_fn(loss_fn=lambda p: self.total_loss(batch, p))
        return self.replace(network=new_network), info

    @jax.jit
    def empowerment(self, observations, rng=None):
        """E(s) from M paired draws z ~ m(. | s), s+ ~ q(. | s, z), mean bound. [B, D] -> [B]."""
        rng = rng if rng is not None else self.rng
        B = observations.shape[0]
        M = self.config['est_num_joints']
        eye = jnp.eye(self.config['num_skills'])
        k_z, k_x = jax.random.split(rng)

        s = jnp.repeat(observations, M, axis=0)
        logm = jax.nn.log_softmax(self.network.select('clf')(s), axis=-1)
        z = jax.random.categorical(k_z, logm, axis=-1)
        heads = self.network.select('q')(jnp.concatenate([s, eye[z]], axis=-1))
        deltas = jax.vmap(mdn_sample)(jax.random.split(k_x, s.shape[0]), *heads)
        b = (eye[z].T * self._bracket_allz(s, deltas)).sum(0)
        return b.reshape(B, M).mean(axis=1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        obs_dim = ex_observations.shape[-1]
        k = config['num_skills']
        layer_norm = config['layer_norm']

        enc_def = MLP((*config['enc_hidden_dims'], k), activate_final=False, layer_norm=layer_norm)
        q_def = MDN(obs_dim, config['num_components'], tuple(config['mdn_hidden_dims']), layer_norm=layer_norm)
        clf_def = MLP((*config['clf_hidden_dims'], k), activate_final=False, layer_norm=layer_norm)

        batch_shape = ex_observations.shape[:-1]
        network_def = ModuleDict(dict(enc=enc_def, q=q_def, clf=clf_def))
        network_params = network_def.init(
            init_rng,
            enc=(jnp.zeros((*batch_shape, obs_dim * 2)),),
            q=(jnp.zeros((*batch_shape, obs_dim + k)),),
            clf=(ex_observations,),
        )['params']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='empowerment_dads',
            lr=3e-4,
            batch_size=1024,
            num_skills=15,  # k
            num_components=8,  # mixture components of q
            mdn_hidden_dims=(512, 512, 512),
            clf_hidden_dims=(512, 512, 512),
            enc_hidden_dims=(512, 512, 512),
            layer_norm=True,
            est_num_joints=16,  # M draws per state in empowerment()
            kl_coef=0.0,  # lambda_kl, weight on KL(Unif[k] || skill usage)
            discount=0.99,
            # s+ is a geometrically-sampled future state of the same trajectory
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
        )
    )
