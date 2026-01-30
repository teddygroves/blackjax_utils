import jax
import jax.numpy as jnp
import pytest
from blackjax_utils.mcmc import get_init_params, run_nuts


def test_get_init_params_no_jitter():
    key = jax.random.PRNGKey(0)
    base_params = {"x": jnp.array([1.0, 2.0])}

    # Test with sd=None
    params_none = get_init_params(key, base_params, sd=None)
    assert jnp.array_equal(params_none["x"], base_params["x"])

    # Test with sd=0
    params_zero = get_init_params(key, base_params, sd=0.0)
    assert jnp.array_equal(params_zero["x"], base_params["x"])


def test_get_init_params_jitter():
    key = jax.random.PRNGKey(0)
    base_params = {"x": jnp.array([1.0, 2.0])}
    sd = 0.1

    params_jitter = get_init_params(key, base_params, sd=sd)

    # Check values are different
    assert not jnp.array_equal(params_jitter["x"], base_params["x"])

    # Check shape is preserved
    assert params_jitter["x"].shape == base_params["x"].shape


def log_density_fn(params):
    return -0.5 * jnp.sum(params["x"] ** 2)


def test_run_nuts_pmap():
    # Scenario: n_chain <= device_count (usually 1 on standard CPU)
    # This triggers the pmap path if n_chain is small enough relative to devices.
    # Assuming 1 CPU device available.
    n_chain = 1
    if jax.local_device_count() < n_chain:
        pytest.skip("Not enough devices for pmap test with n_chain=1")

    key = jax.random.PRNGKey(1)
    init_params = {"x": jnp.array([10.0])}  # Start far from 0

    states, info = run_nuts(
        key=key,
        log_posterior=log_density_fn,
        init_params=init_params,
        init_sd=1.0,
        n_chain=n_chain,
        n_warmup=200,
        n_sample=1000,
        max_num_doublings=5,
    )

    samples = states.position["x"]
    # Expected shape: (n_chain, n_sample, dim)
    assert samples.shape == (n_chain, 1000, 1)

    # Check statistics (standard normal)
    mean = jnp.mean(samples)
    std = jnp.std(samples)

    # Allow some tolerance for stochasticity
    assert jnp.abs(mean) < 0.2
    assert jnp.abs(std - 1.0) < 0.2


def test_run_nuts_vmap():
    # Scenario: n_chain > device_count
    # On a 1-device machine, n_chain=2 guarantees the vmap path.
    n_chain = jax.local_device_count() + 1

    key = jax.random.PRNGKey(2)
    init_params = {"x": jnp.array([10.0])}

    states, info = run_nuts(
        key=key,
        log_posterior=log_density_fn,
        init_params=init_params,
        init_sd=1.0,
        n_chain=n_chain,
        n_warmup=200,
        n_sample=1000,
        max_num_doublings=5,
    )

    samples = states.position["x"]
    assert samples.shape == (n_chain, 1000, 1)

    # Check statistics across all chains
    mean = jnp.mean(samples)
    std = jnp.std(samples)

    assert jnp.abs(mean) < 0.2
    assert jnp.abs(std - 1.0) < 0.2
