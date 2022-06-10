import jax
import jax.numpy as jnp

def grf(
        seed: int = 1234,
        xdim: int = 256,
        ydim: int = 256,
        sigma: float = 0.1,
        rho: float = 0.1,
        n: int = 1,
        ):

    """
    Variables seeting for random 

    seed : random seed
    sigma : scale(?)
    rho : smoothness(?)
    """

    """
    Isotropic covariance Gaussian random field with exponentiated quadratic (i.e. RBF) covariance function.

    The first dimension is the batch dimension.
    """
    rng = jax.random.PRNGKey(seed)
    fx, fy = jnp.meshgrid(
        jnp.fft.fftfreq(xdim) * xdim,
        jnp.fft.rfftfreq(ydim) * ydim,
        indexing = 'ij')
    nfx, nfy = fx.shape
    fnorm = jnp.sqrt(fx**2 + fy**2)
    power = jnp.exp(-(fnorm**2/rho))
    gain = jnp.sqrt(sigma**2 * power/power.sum())  # Lazy not calculating normalisation

    noise = (
        jax.random.normal(rng, (n, nfx, nfy))
        + jax.random.normal(rng, (n, nfx, nfy)) * 1j
    )
    noise = noise.at[...,0].set(jnp.abs(noise[..., 0]))
    ## TODO: This is the rbf kernel; Matern kernel has more plausible smoothness.
    ## Matern 3/2 PSD is
    #(18 * jnp.sqrt(3)* jnp.pi * sigma**2)/((4 * k^2 * jnp.pi**2 + 3/(rho**2))^(5/2) rho^3)
    field = jnp.fft.irfft2(noise * gain, (xdim, ydim), norm="forward")
    return field
