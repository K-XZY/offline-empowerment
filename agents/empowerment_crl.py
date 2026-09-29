"""CF-Empowerment baseline: action empowerment I(s+; a | s) from two contrastive critics + distillation.

  action critic:    f1(s, a, s+) = phi(s, a) . psi(s+) / sqrt(d)
  no-action critic: f2(s, s+)    = phi'(s)   . psi(s+) / sqrt(d)

Both use binary NCE with in-batch negatives (as in CRL) and share psi. At the optimum f1 - f2 is the pointwise
conditional MI log p(s+ | s, a) - log p(s+ | s). After `crl_pretrain_steps` the critics are frozen and E_theta(s)
is regressed onto f1 - f2, so E_theta(s) -> I(s+; a | s).
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


class EmpowermentCRLCritic(nn.Module):
    """phi(s, a), phi'(s) and the shared psi(s+)."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True

    def setup(self):
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.phi_prime = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions, future_states):
        return (
            self.phi(jnp.concatenate([observations, actions], axis=-1)),
            self.phi_prime(observations),
            self.psi(future_states),
        )


class EmpowermentNet(nn.Module):
    """Scalar E_theta(s)."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations):
        out = MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)(observations)
        return out.squeeze(-1)


class EmpowermentCRLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params):
        batch_size = batch['observations'].shape[0]

        phi, phi_prime, psi = self.network.select('critic')(
            batch['observations'], batch['actions'], batch['value_goals'], params=grad_params
        )
        scale = jnp.sqrt(phi.shape[-1])
        logits_action = jnp.einsum('ik,jk->ij', phi, psi) / scale
        logits_state = jnp.einsum('ik,jk->ij', phi_prime, psi) / scale

        I = jnp.eye(batch_size)
        loss_action = optax.sigmoid_binary_cross_entropy(logits=logits_action, labels=I).mean()
        loss_state = optax.sigmoid_binary_cross_entropy(logits=logits_state, labels=I).mean()
        loss = loss_action + loss_state

        def diag_stats(logits):
            pos = jnp.sum(logits * I) / jnp.sum(I)
            neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
            cat_acc = jnp.mean(jnp.argmax(logits, axis=1) == jnp.arange(batch_size))
            bin_acc = jnp.mean((logits > 0) == I)
            return pos, neg, cat_acc, bin_acc

        a_pos, a_neg, a_cat, a_bin = diag_stats(logits_action)
        s_pos, s_neg, s_cat, s_bin = diag_stats(logits_state)
        return loss, {
            'contrastive_loss': loss,
            'loss_action': loss_action,
            'loss_state': loss_state,
            'logits_action_pos': a_pos,
            'logits_action_neg': a_neg,
            'cat_acc_action': a_cat,
            'bin_acc_action': a_bin,
            'logits_state_pos': s_pos,
            'logits_state_neg': s_neg,
            'cat_acc_state': s_cat,
            'bin_acc_state': s_bin,
        }

    def distill_loss(self, batch, grad_params):
        """MSE of E_theta(s) onto T(s, a, s+) = (phi(s, a) - phi'(s)) . psi(s+) / sqrt(d), critics frozen."""
        phi, phi_prime, psi = self.network.select('critic')(
            batch['observations'], batch['actions'], batch['value_goals']
        )
        phi, phi_prime, psi = jax.lax.stop_gradient((phi, phi_prime, psi))
        target = jnp.sum((phi - phi_prime) * psi, axis=-1) / jnp.sqrt(phi.shape[-1])

        pred = self.network.select('empowerment_net')(batch['observations'], params=grad_params)
        loss = jnp.mean((pred - target) ** 2)
        return loss, {
            'distill_loss': loss,
            'target_mean': target.mean(),
            'target_min': target.min(),
            'target_max': target.max(),
            'pred_mean': pred.mean(),
            'pred_min': pred.min(),
            'pred_max': pred.max(),
        }

    def empowerment(self, observations, rng=None):
        """E_theta(s) (deterministic, `rng` is unused)."""
        return self.network.select('empowerment_net')(observations)

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        contrastive_loss, c_info = self.contrastive_loss(batch, grad_params)
        info.update({f'crl/{k}': v for k, v in c_info.items()})

        distill_loss, d_info = self.distill_loss(batch, grad_params)
        info.update({f'distill/{k}': v for k, v in d_info.items()})

        # contrastive pretraining first, then distillation only
        in_distill = (self.network.step >= self.config['crl_pretrain_steps']).astype(jnp.float32)
        info['phase/in_distill'] = in_distill

        emp = jax.lax.stop_gradient(self.empowerment(batch['observations']))
        info['empowerment/mean'] = emp.mean()
        info['empowerment/min'] = emp.min()
        info['empowerment/max'] = emp.max()

        return (1.0 - in_distill) * contrastive_loss + in_distill * distill_loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        # keep the critic params fixed once distillation starts (adam momentum would still move them otherwise)
        old_critic = self.network.params['modules_critic']
        in_distill = self.network.step >= self.config['crl_pretrain_steps']

        new_network, info = self.network.apply_loss_fn(loss_fn=lambda p: self.total_loss(batch, p, rng=rng))

        frozen_critic = jax.tree_util.tree_map(
            lambda old, new: jnp.where(in_distill, old, new), old_critic, new_network.params['modules_critic']
        )
        new_network = new_network.replace(params={**new_network.params, 'modules_critic': frozen_critic})
        return self.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        critic_def = EmpowermentCRLCritic(
            hidden_dims=tuple(config['value_hidden_dims']),
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
        )
        empowerment_def = EmpowermentNet(
            hidden_dims=tuple(config['empowerment_hidden_dims']),
            layer_norm=config['layer_norm'],
        )

        network_def = ModuleDict(dict(critic=critic_def, empowerment_net=empowerment_def))
        network_params = network_def.init(
            init_rng,
            critic=(ex_observations, ex_actions, ex_observations),
            empowerment_net=(ex_observations,),
        )['params']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='empowerment_crl',
            lr=1e-3,
            batch_size=512,
            value_hidden_dims=(256, 256),  # phi, phi', psi
            latent_dim=256,
            layer_norm=True,
            empowerment_hidden_dims=(256, 256),  # E_theta
            crl_pretrain_steps=500_000,  # contrastive steps before switching to distillation
            discount=0.99,
            # s+ is a geometrically-sampled future state of the same trajectory
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
        )
    )
