"""DV-Empowerment baseline: same pi_bc, f_dyn and critic T as CF-FlowBC, but the Donsker-Varadhan bound

    J = T(s+, a, s) - log (1/N) sum_i exp T(s+_{sigma(i)}, a_i, s),

with a_i ~ pi_bc(. | s), s+_i ~ f_dyn(. | s, a_i) and sigma a cyclic derangement, so every (a_i, s+_{sigma(i)}) is a
sample from the product of marginals. The critic maximizes J; the flows get no gradient through the negatives.
"""

import jax
import jax.numpy as jnp
import ml_collections

from agents.empowerment_crl_flowbc import EmpowermentCRLFlowBCAgent


class EmpowermentDVAgent(EmpowermentCRLFlowBCAgent):
    def _neg_term(self, s, rng, grad_params):
        """log-mean-exp of T over N deranged product-of-marginals negatives at s -> [B]."""
        B, D = s.shape[0], s.shape[-1]
        N = self.config['num_negatives']
        k_a, k_sp = jax.random.split(rng)
        s_rep = jnp.broadcast_to(s[:, None, :], (B, N, D))
        a_neg = jax.lax.stop_gradient(self._flow_sample('bc_flow', s_rep, k_a, self.config['action_dim'], clip=1.0))
        cond = jnp.concatenate([s_rep, a_neg], axis=-1)
        sp_neg = jax.lax.stop_gradient(self._flow_sample('dyn_flow', cond, k_sp, self.config['obs_dim']))
        # pair a_i with s+_{(i + 1) mod N}
        sp_deranged = jnp.roll(sp_neg, shift=-1, axis=1)
        neg_logits = self.network.select('critic')(sp_deranged, a_neg, s_rep, params=grad_params)  # [B, N]
        return jax.nn.logsumexp(neg_logits, axis=1) - jnp.log(N)

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        bc_loss, dyn_loss, in_critic, k_neg = self._pretrain_losses(batch, grad_params, rng)
        s, a, sp = batch['observations'], batch['actions'], batch['value_goals']

        def critic_branch(_):
            pos_logit = self.network.select('critic')(sp, a, s, params=grad_params)
            return -jnp.mean(pos_logit - self._neg_term(s, k_neg, grad_params))

        critic_loss = jax.lax.cond(in_critic, critic_branch, lambda _: jnp.float32(0.0), None)
        info = {
            'bc_loss': bc_loss,
            'dyn_loss': dyn_loss,
            'critic_loss': critic_loss,
            'mi_estimate': jnp.where(in_critic, -critic_loss, 0.0),
            'phase/in_critic': in_critic.astype(jnp.float32),
        }
        return bc_loss + dyn_loss + critic_loss, info

    @jax.jit
    def empowerment(self, observations, rng=None):
        """E(s) = mean_k T(s+_k, a_k, s) - logmeanexp over the deranged negatives. [B, D] -> [B]."""
        rng = rng if rng is not None else self.rng
        B, D = observations.shape[0], observations.shape[-1]
        K = self.config['est_num_joints']
        k_a, k_sp, k_neg = jax.random.split(rng, 3)

        s_rep = jnp.broadcast_to(observations[:, None, :], (B, K, D))
        a_pos = self.sample_bc(s_rep, k_a)
        sp_pos = self.sample_dyn(s_rep, a_pos, k_sp)
        pos_term = self.network.select('critic')(sp_pos, a_pos, s_rep).mean(axis=1)
        return pos_term - self._neg_term(observations, k_neg, None)


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='empowerment_dv',
            lr=3e-4,
            batch_size=256,
            flow_hidden_dims=(512, 512, 512),
            critic_hidden_dims=(512, 512, 512),
            layer_norm=True,
            flow_steps=8,  # euler steps for flow sampling
            num_negatives=7,  # N deranged negatives
            pretrain_steps=100000,  # critic starts training at this train step (pi_bc / f_dyn only before)
            est_num_joints=16,  # joints per state in empowerment()
            discount=0.99,
            # s+ is a geometrically-sampled future state of the same trajectory
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
        )
    )
