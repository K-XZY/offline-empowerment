"""OPAL VAE skills (Ajay et al. 2021), the offline skill-pretraining stage of SUPE (Wilcoxson et al. 2025).

Ported from the SUPE repo (supe/pretraining/opal.py, configs/opal_config.py, run_opal.py). Given a length-c window
tau = (s_{1:c}, a_{1:c}):
  posterior q(z | tau): BiGRU sequence encoder -> (mean, log_std); log_std is used raw and std = exp(0.5 log_std)
  prior p(z | s_1):     gaussian mlp head on the first state (log_std clipped to [-20, 2])
  decoder pi(a | s, z): gaussian mlp head, the low-level skill policy
loss = -E_q[log pi(a_{1:c} | s_{1:c}, z)] + kl_coef * KL(q(z | tau) || p(z | s_1)).

Windows are clamped to the trajectory end with a `seq_mask` (the SUPE ChunkDataset only samples full windows), so the
recon term is a masked mean over real steps, identical to the source when a window has no padding.
"""

from functools import partial
from typing import Any, Callable, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field


def default_init(scale: Optional[float] = 1.0):
    """variance_scaling(scale, 'fan_avg', 'uniform') -- SUPE's `default_init`."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class MLP(nn.Module):
    """supe/networks/mlp.py"""

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    use_pnorm: bool = False
    default_init: nn.initializers.Initializer = nn.initializers.xavier_uniform

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: float = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=self.default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=self.default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        if self.use_pnorm:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-10)
        return x


# as in the source, OPAL binds MLP's init to `default_init`
MLP = partial(MLP, default_init=default_init)


class SimpleGRU(nn.Module):
    hidden_size: int

    def setup(self):
        self.gru = nn.GRUCell(features=self.hidden_size)

    @partial(
        nn.transforms.scan,
        variable_broadcast='params',
        in_axes=1,
        out_axes=1,
        split_rngs={'params': False},
    )
    def __call__(self, carry, x):
        return self.gru(carry, x)


class SimpleBiGRU(nn.Module):
    hidden_size: int

    def setup(self):
        self.forward_gru = SimpleGRU(self.hidden_size)
        self.backward_gru = SimpleGRU(self.hidden_size)

    def __call__(self, embedded_inputs):
        shape = embedded_inputs[:, 0].shape

        initial_state = self.forward_gru.gru.initialize_carry(jax.random.key(0), shape)
        _, forward_outputs = self.forward_gru(initial_state, embedded_inputs)

        reversed_inputs = embedded_inputs[:, ::-1, :]
        initial_state = self.backward_gru.gru.initialize_carry(jax.random.key(0), shape)
        _, backward_outputs = self.backward_gru(initial_state, reversed_inputs)
        backward_outputs = backward_outputs[:, ::-1, :]

        return jnp.concatenate([forward_outputs, backward_outputs], -1)


class SeqEncoder(nn.Module):
    """q(z | tau): per-step obs mlp -> concat action -> BiGRUs -> concat over time -> linear to (mean, log_std)."""

    num_recur_layers: int = 2
    output_dim: int = 2
    recur_output: str = 'concat'
    hidden_size: int = 256

    def setup(self) -> None:
        self.obs_mlp = MLP([self.hidden_size, self.hidden_size], activate_final=True)
        self.recurs = [SimpleBiGRU(self.hidden_size) for _ in range(self.num_recur_layers)]
        self.projection = MLP([self.output_dim], activate_final=False)

    def __call__(self, seq_observations: jnp.ndarray, seq_actions: jnp.ndarray):
        B, C, D = seq_observations.shape
        observations = jnp.reshape(seq_observations, (B * C, D))
        outputs = jnp.reshape(self.obs_mlp(observations), (B, C, -1))
        outputs = jnp.concatenate([outputs, seq_actions], axis=-1)

        for recur in self.recurs:
            outputs = recur(outputs)
        if self.recur_output == 'concat':
            outputs = jnp.reshape(outputs, (B, -1))
        else:
            outputs = outputs[:, -1]
        return self.projection(outputs)


class GaussianModule(nn.Module):
    """Diagonal gaussian mlp head, used for the prior and the decoder."""

    hidden_dims: Sequence[int]
    output_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, temperature: float = 1.0) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(inputs)
        means = nn.Dense(self.output_dim, kernel_init=default_init(self.final_fc_init_scale))(outputs)
        log_stds = nn.Dense(self.output_dim, kernel_init=default_init(self.final_fc_init_scale))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        return distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)


class OPALAgent(flax.struct.PyTreeNode):
    """Networks: encoder (q(z | tau)), prior (p(z | s_1)), decoder (pi(a | s, z))."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def vae_loss(self, batch, grad_params, rng):
        obs_seq = batch['observations_seq']  # [B, C, obs_dim]
        act_seq = batch['actions_seq']  # [B, C, act_dim]
        seq_mask = batch['seq_mask']  # [B, C]
        B, C = seq_mask.shape
        skill_dim = self.config['skill_dim']

        enc_out = self.network.select('encoder')(obs_seq, act_seq, params=grad_params)
        means = enc_out[..., :skill_dim]
        log_stds = enc_out[..., skill_dim:]
        stds = jnp.exp(0.5 * log_stds)
        posteriors = distrax.MultivariateNormalDiag(loc=means, scale_diag=stds)

        priors = self.network.select('prior')(obs_seq[:, 0], params=grad_params)

        # one reparameterized skill for the whole window
        zs = means + stds * jax.random.normal(rng, means.shape)
        zs = jnp.broadcast_to(zs[:, None, :], (B, C, skill_dim))
        szs = jnp.concatenate([obs_seq, zs], axis=-1)
        recon_action_dists = self.network.select('decoder')(szs, params=grad_params)

        recon_logprob = recon_action_dists.log_prob(act_seq)
        recon_loss = -(recon_logprob * seq_mask).sum() / jnp.maximum(seq_mask.sum(), 1.0)
        kl_loss = posteriors.kl_divergence(priors).mean()
        total_loss = recon_loss + self.config['kl_coef'] * kl_loss

        return total_loss, {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss,
            'prior_mean': priors.loc.mean(),
            'prior_std': priors.scale_diag.mean(),
            'posterior_mean': posteriors.loc.mean(),
            'posterior_std': posteriors.scale_diag.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        return self.vae_loss(batch, grad_params, rng)

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=lambda p: self.total_loss(batch, p, rng=rng))
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions_with_skill(self, observations, skills, seed=None, temperature=1.0):
        """a ~ pi(a | s, z) for a fixed skill z."""
        if seed is None:
            seed = self.rng
        single_obs = observations.ndim == 1
        obs = observations[None] if single_obs else observations

        skills = skills[None, ...] if skills.ndim == 1 else skills
        skills = jnp.broadcast_to(skills, (obs.shape[0], skills.shape[-1]))

        szs = jnp.concatenate([obs, skills], axis=-1)
        actions = self.network.select('decoder')(szs, temperature).sample(seed=seed)
        actions = jnp.clip(actions, -1.0, 1.0)
        return actions[0] if single_obs else actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        assert config.get('latent_type', 'continuous') == 'continuous', 'only the continuous (SUPE) OPAL is included'
        assert int(config['sequence_length']) == int(config['chunk_size'])

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        config = dict(config)
        skill_dim = config['skill_dim']
        action_dim = ex_actions.shape[-1]

        B = ex_observations.shape[0]
        C = int(config['chunk_size'])
        ex_obs_seq = jnp.broadcast_to(ex_observations[:, None], (B, C) + ex_observations.shape[1:])
        ex_act_seq = jnp.broadcast_to(ex_actions[:, None], (B, C) + ex_actions.shape[1:])
        ex_sz = jnp.concatenate([ex_observations, jnp.zeros((B, skill_dim))], axis=-1)

        encoder_def = SeqEncoder(
            num_recur_layers=2,
            output_dim=skill_dim * 2,
            recur_output='concat',
            hidden_size=config['vae_encoder_hidden_size'],
        )
        prior_def = GaussianModule(hidden_dims=tuple(config['vae_hidden_dims']), output_dim=skill_dim)
        decoder_def = GaussianModule(hidden_dims=tuple(config['vae_hidden_dims']), output_dim=action_dim)

        network_def = ModuleDict(dict(encoder=encoder_def, prior=prior_def, decoder=decoder_def))
        network_params = network_def.init(
            init_rng,
            encoder=(ex_obs_seq, ex_act_seq),
            prior=(ex_observations,),
            decoder=(ex_sz,),
        )['params']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='opal',
            lr=3e-4,
            batch_size=128,
            latent_type='continuous',
            skill_dim=8,  # latent z dim
            kl_coef=0.1,
            chunk_size=10,  # skill horizon H (must match the online skill_commitment_k)
            vae_hidden_dims=(512, 512, 512),  # prior + decoder
            vae_encoder_hidden_size=512,  # BiGRU / obs mlp width
            # windows of `sequence_length` (== chunk_size) steps from the dataset
            dataset_class='SequenceDataset',
            sequence_length=10,
        )
    )
