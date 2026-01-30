from functools import partial
from typing import Any, Callable

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


def inference_loop(
    key: PRNGKeyArray,
    tuned_params: dict[str, Any],
    initial_state: PyTree,
    num_samples: int,
    log_posterior: Callable,
    **static_params: Any,
) -> tuple[PyTree, PyTree]:
    """Run a sampling loop.

    Args:
        key: A JAX PRNG key.
        tuned_params: Dict of tuned NUTS parameters (step_size, inverse_mass_matrix, etc.)
        initial_state: Initial state for sampling from warmup.
        num_samples: Number of samples to draw.
        log_posterior: The log-probability density function of the target distribution.
        **static_params: Static algorithm parameters (e.g., max_num_doublings) that
            are not tuned during warmup.

    Returns:
        A tuple containing (states, info) where states is the tree of samples
        and info contains sampling diagnostics.
    """
    # Merge static params with tuned params for the kernel
    all_params = {**tuned_params, **static_params}
    kernel = blackjax.nuts(log_posterior, **all_params).step

    @jax.jit
    def one_step(state: Any, rng_key: PRNGKeyArray) -> tuple[Any, tuple[Any, Any]]:
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(key, num_samples)
    _, (states, info) = jax.lax.scan(one_step, initial_state, keys)

    return states, info


def run_chain(
    key: PRNGKeyArray,
    init_params: PyTree,
    target_density: Callable,
    warmup_kwargs: dict[str, Any],
    n_warmup: int,
    n_sample: int,
) -> tuple[PyTree, PyTree]:
    """Run warmup and sampling for a single chain.

    This function runs the full MCMC workflow for a single chain: warmup
    (window adaptation) followed by NUTS sampling.

    Args:
        key: A JAX PRNG key.
        init_params: Initial parameter values.
        target_density: The log-density function to sample from.
        warmup_kwargs: Additional arguments passed to `blackjax.window_adaptation`.
        n_warmup: Number of warmup (adaptation) steps.
        n_sample: Number of sampling steps.

    Returns:
        A tuple containing (states, info) where states is the tree of samples
        and info contains sampling diagnostics.
    """
    warmup_key, sample_key = jax.random.split(key)
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        target_density,
        **warmup_kwargs,
    )
    (warmed_up_state, tuned_params), _ = warmup.run(
        warmup_key,
        init_params,
        n_warmup,  # type: ignore
    )
    sample_loop = partial(
        inference_loop,
        num_samples=n_sample,
        log_posterior=target_density,
        **tuned_params,
    )
    return sample_loop(sample_key, tuned_params, warmed_up_state)


def run_nuts(
    key: PRNGKeyArray,
    log_posterior: Callable,
    init_params: PyTree,
    init_sd: PyTree | None = None,
    n_chain: int = 4,
    n_warmup: int = 500,
    n_sample: int = 500,
    **warmup_kwargs: Any,
) -> tuple[PyTree, PyTree]:
    """Run NUTS sampling with automatic parallelization across multiple chains.

    This function coordinates the full MCMC workflow: initialization, warmup
    (adaptation), and sampling. It automatically chooses between `jax.pmap`
    (multi-device parallelism) and `jax.vmap` (vectorization on a single device)
    based on the number of available devices and the requested number of chains.

    Args:
        key: A JAX PRNG key.
        log_posterior: The log-probability density function of the target distribution.
        init_params: Initial values for the parameters. Will be jittered by `init_sd`.
        init_sd: Standard deviation for jittering the initial parameters. If None,
            start exactly at `init_params`.
        n_chain: Number of MCMC chains to run.
        n_warmup: Number of warmup (adaptation) steps per chain.
        n_sample: Number of sampling steps per chain.
        **warmup_kwargs: Additional keyword arguments passed to `blackjax.window_adaptation`,
            such as `max_num_doublings`, `is_mass_matrix_diagonal`, etc.

    Returns:
        A tuple containing (states, info) where:
        - states: Tree of posterior samples with shape (n_chain, n_sample, ...)
        - info: Dictionary with sampling diagnostics (e.g., divergence info)
    """
    map_func = jax.pmap if jax.local_device_count() >= n_chain else jax.vmap
    sample_keys, init_keys = jax.random.split(key, (2, n_chain))
    get_init_params_vmap = jax.vmap(get_init_params, in_axes=(0, None, None))
    init_params = get_init_params_vmap(init_keys, init_params, init_sd)
    run_this_chain = partial(
        run_chain,
        target_density=log_posterior,
        warmup_kwargs=warmup_kwargs,
        n_warmup=n_warmup,
        n_sample=n_sample,
    )
    run_these_chains = map_func(run_this_chain, in_axes=(0, 0))
    return run_these_chains(sample_keys, init_params)
