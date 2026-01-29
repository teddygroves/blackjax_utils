from functools import partial
from typing import Any, Callable, Dict, Tuple

import blackjax
import jax
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree


def get_init_params(key: PRNGKeyArray, base: PyTree, sd: PyTree | None) -> PyTree:
    """Initialize parameters by adding jitter to a base value.

    Args:
        key: A JAX PRNG key.
        base: The base parameters (PyTree) to jitter around.
        sd: The standard deviation of the jitter. If None, no jitter is applied
            (sd is treated as zero).

    Returns:
        The jittered parameters with the same structure as `base`.
    """

    def jitter_leaf(key: PRNGKeyArray, base_leaf: Any, sd_leaf: Any) -> Any:
        return base_leaf + jax.random.normal(key, shape=base_leaf.shape) * sd_leaf

    flat_means, treedef = jax.tree.flatten(base)
    keys = jax.random.split(key, num=len(flat_means))
    keytree = jax.tree.unflatten(treedef, keys)
    if sd is None:
        sd = jax.tree.map(jnp.zeros_like, base)
    elif isinstance(sd, (int, float)) or (hasattr(sd, "shape") and sd.shape == ()):
        sd_scalar = sd
        sd = jax.tree.map(lambda _: sd_scalar, base)
    return jax.tree.map(jitter_leaf, keytree, base, sd)


def run_warmup(
    key: PRNGKeyArray,
    params: PyTree,
    target_density: Callable,
    draws: int,
    warmup_kwargs: Dict[str, Any],
) -> Tuple[PyTree, Dict[str, Any]]:
    """Run the NUTS window adaptation warmup.

    Args:
        key: A JAX PRNG key.
        params: Initial parameters.
        target_density: The log-density function to sample from.
        draws: Number of warmup steps.
        warmup_kwargs: Additional arguments passed to `blackjax.window_adaptation`.

    Returns:
        A tuple containing the final warmup state and the tuned NUTS parameters.
    """
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        target_density,
        **warmup_kwargs,
    )
    (initial_states, tuned_params), _ = warmup.run(key, params, draws)  # type: ignore
    return initial_states, tuned_params


def inference_loop(
    key: PRNGKeyArray,
    tuned_params: dict[str, Any],
    initial_state: PyTree,
    num_samples: int,
    log_posterior: Callable,
) -> Tuple[PyTree, PyTree]:
    """Run a sampling loop.

    Args:
        key: A JAX PRNG key.
        kernel: a blackjax kernel function
        num_samples: Number of samples to draw.

    Returns:
        A tuple containing the tree of samples and the sampling info/diagnostics.
    """
    kernel = blackjax.nuts(log_posterior, **tuned_params).step

    @jax.jit
    def one_step(state: Any, rng_key: PRNGKeyArray) -> Tuple[Any, Tuple[Any, Any]]:
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(key, num_samples)
    _, (states, info) = jax.lax.scan(one_step, initial_state, keys)

    return states, info


def run_nuts(
    key: PRNGKeyArray,
    log_posterior: Callable,
    init_params: PyTree,
    init_sd: PyTree | None = None,
    n_chain: int = 4,
    n_warmup: int = 500,
    n_sample: int = 500,
    **warmup_kwargs: Any,
) -> Tuple[PyTree, PyTree]:
    """Run NUTS sampling with automatic parallelization.

    This function coordinates the full MCMC workflow: initialization, warmup
    (adaptation), and sampling. It automatically chooses between `jax.pmap`
    (multi-device parallelism) and `jax.vmap` (vectorization on a single device)
    based on the number of available devices and the requested number of chains.

    Args:
        key: A JAX PRNG key.
        log_posterior: The log-probability density function of the target distribution.
        init_params: Initial values for the parameters.
        init_sd: Standard deviation for jittering the initial parameters. If None,
            start exactly at `init_params`.
        n_chain: Number of MCMC chains to run.
        n_warmup: Number of warmup (adaptation) steps.
        n_sample: Number of sampling steps per chain.
        **warmup_kwargs: Additional keyword arguments passed to the warmup adaptation.

    Returns:
        A tuple containing the posterior samples and sampling diagnostics (info).
    """
    inference_loop_concrete = partial(
        inference_loop,
        num_samples=n_sample,
        log_posterior=log_posterior,
    )
    run_warmup_concrete = partial(
        run_warmup,
        target_density=log_posterior,
        draws=n_warmup,
        warmup_kwargs=warmup_kwargs,
    )
    map_func = jax.pmap if jax.local_device_count() >= n_chain else jax.vmap
    warmup_func = map_func(run_warmup_concrete, in_axes=(0, 0))
    sample_func = map_func(inference_loop_concrete, in_axes=(0, 0, 0))
    sample_key, warmup_key, init_key = jax.random.split(key, 3)
    warmup_keys = jax.random.split(warmup_key, n_chain)
    sample_keys = jax.random.split(sample_key, n_chain)
    init_keys = jax.random.split(init_key, n_chain)
    get_init_params_vmap = jax.vmap(get_init_params, in_axes=(0, None, None))
    init_params = get_init_params_vmap(init_keys, init_params, init_sd)
    initial_states, tuned_params = warmup_func(warmup_keys, init_params)
    states, info = sample_func(sample_keys, tuned_params, initial_states)
    return states, info
