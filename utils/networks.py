from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    """Multi-layer perceptron."""

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and std (used for the RND input / reward normalization)."""

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        return jnp.clip(batch, -self.clip_max, self.clip_max)

    def update_masked(self, batch, mask):
        """Parallel mean/var merge over the rows where `mask` is 1 (no-op if none are)."""
        mask = jnp.asarray(mask, dtype=jnp.float32)
        batch_count = mask.sum()
        w = mask.reshape((-1,) + (1,) * (batch.ndim - 1))
        safe_batch = jnp.maximum(batch_count, 1.0)
        batch_mean = (batch * w).sum(axis=0) / safe_batch
        batch_var = (jnp.square(batch - batch_mean) * w).sum(axis=0) / safe_batch

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        safe_total = jnp.maximum(total_count, 1.0)
        new_mean = self.mean + delta * batch_count / safe_total
        m_2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / safe_total
        new_var = m_2 / safe_total

        empty = batch_count == 0
        return self.replace(
            mean=jnp.where(empty, self.mean, new_mean),
            var=jnp.where(empty, self.var, new_var),
            count=total_count,
        )


class RNDEmbedding(nn.Module):
    """RND embedding (Burda et al. 2019), same arch for the fixed target and the predictor.

    relu mlp + linear head, orthogonal init with gain sqrt(2) like the official code.
    """

    hidden_dims: Sequence[int]
    rep_dim: int

    @nn.compact
    def __call__(self, x):
        init = nn.initializers.orthogonal(scale=2.0**0.5)
        for size in self.hidden_dims:
            x = nn.relu(nn.Dense(size, kernel_init=init)(x))
        return nn.Dense(self.rep_dim, kernel_init=init)(x)


class GCActor(nn.Module):
    """Goal-conditioned gaussian actor."""

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        elif not self.const_std:
            self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(self, observations, goals=None, temperature=1.0):
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        outputs = self.actor_net(jnp.concatenate(inputs, axis=-1))

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        elif self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = self.log_stds
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))
        return distribution


class GCValue(nn.Module):
    """(Goal-conditioned) value / critic, optionally a 2-head ensemble."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        self.value_net = mlp_module((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals=None, actions=None):
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        return self.value_net(jnp.concatenate(inputs, axis=-1)).squeeze(-1)


class GCBilinearValue(nn.Module):
    """Bilinear critic Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d)."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        phi_inputs = observations if actions is None else jnp.concatenate([observations, actions], axis=-1)
        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        return v
