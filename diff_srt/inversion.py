"""Gradient-based inversion and sensitivity analysis for the JAX SRT model."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .forward import brf_topo
from .jax_srt import brf_flat_sphere

PARAM_NAMES = ("H", "cover", "d_L", "slope", "aspect")
DEFAULT_BOUNDS = np.array(
    [
        [1.0, 15.0],    # H
        [0.05, 0.95],   # cover
        [0.1, 3.0],     # d_L
        [0.0, 60.0],    # slope
        [0.0, 360.0],   # aspect
    ],
    dtype=float,
)


@dataclass
class Observation:
    sza: float
    saa: float
    vza: np.ndarray
    vaa: np.ndarray
    brf: np.ndarray
    sigma: float = 0.01


@dataclass
class ModelConfig:
    tree_type: int = 4
    lad_type: int = 6
    diam: float = 5.0
    ro_soil: float = 0.3379
    ro_L: float = 0.4746
    tau_L: float = 0.4843
    nz: int = 40
    # Hot-spot parameter (<=0 disables it).
    hotspot: float = 0.0
    # Use the full validated SRT solver (spherical crowns, flat or sloping
    # ground) instead of the reduced direct+single-scattering model.
    full_flat: bool = False
    # Fixed slope geometry for the full solver (slope/aspect must not be free).
    slope: float = 0.0
    aspect: float = 90.0
    # Leaf-orientation rotation (Fortran AlphaSlope, degrees); None -> slope.
    alpha_slope: float | None = None


def encode(x, lo, hi):
    u = (x - lo) / (hi - lo)
    u = jnp.clip(u, 1e-4, 1.0 - 1e-4)
    return jnp.log(u / (1.0 - u))


def decode(z, lo, hi):
    return lo + (hi - lo) * jax.nn.sigmoid(z)


# --------------------------------------------------------------------------
# LAI parameterization
# --------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_GL_X = jnp.asarray(0.5 * (_GL_X + 1.0))
_GL_W = jnp.asarray(0.5 * _GL_W)


def shape_factor(cover, tree_type=4):
    """Normalized LAI factor Lambda(cover) = int_0^1 p(u) du (u = z/H)."""
    u = _GL_X
    if tree_type == 1:
        return cover + 0.0 * jnp.sum(_GL_W)
    if tree_type == 2:
        p = 1.0 - (1.0 - cover) ** (u * u)
    else:  # sphere / ellipsoid use 4u(1-u)
        p = 1.0 - (1.0 - cover) ** (4.0 * u * (1.0 - u))
    return jnp.sum(_GL_W * p)


def lai_from_dl(d_L, H, cover, tree_type=4):
    """LAI = d_L * int_0^H p(z) dz = d_L * H * Lambda(cover)."""
    return d_L * H * shape_factor(cover, tree_type)


def dl_from_lai(LAI, H, cover, tree_type=4):
    return LAI / jnp.maximum(H * shape_factor(cover, tree_type), 1e-6)


def _full_params(z_free, init, free_idx, lo, hi):
    free_idx = jnp.asarray(free_idx)
    vals = decode(z_free, lo[free_idx], hi[free_idx])
    return init.at[free_idx].set(vals)


def predict(
    x_phys,
    obs: Observation,
    model: ModelConfig,
):
    H, cover, d_L, slope, aspect = x_phys
    if model.full_flat:
        return brf_flat_sphere(
            H, cover, d_L, obs.sza, obs.saa, obs.vza, obs.vaa,
            model.ro_L, model.tau_L, model.ro_soil,
            hotspot=model.hotspot, nz=model.nz,
            slope=model.slope, aspect=model.aspect,
            tree_type=model.tree_type, diam=model.diam, lad=model.lad_type,
            alpha_slope=model.alpha_slope,
        )
    brf_plane, _ = brf_topo(
        H,
        cover,
        d_L,
        model.tree_type,
        model.lad_type,
        model.diam,
        jnp.deg2rad(slope),
        obs.sza,
        obs.saa,
        obs.vza,
        obs.vaa,
        slope,
        aspect,
        model.ro_soil,
        model.ro_L,
        model.tau_L,
        nz=model.nz,
        hotspot=model.hotspot,
    )
    return brf_plane


# --------------------------------------------------------------------------
# Compiled-executable cache
# --------------------------------------------------------------------------
from collections import OrderedDict
import hashlib

_JIT_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_JIT_CACHE_LIMIT = 12


def clear_jit_cache():
    """Drop cached compiled inversion/sensitivity executables."""
    _JIT_CACHE.clear()


def _cache(key, build):
    fn = _JIT_CACHE.get(key)
    if fn is not None:
        _JIT_CACHE.move_to_end(key)
        return fn
    fn = build()
    _JIT_CACHE[key] = fn
    while len(_JIT_CACHE) > _JIT_CACHE_LIMIT:
        _JIT_CACHE.popitem(last=False)
    return fn


def _arr_hash(a):
    return hashlib.md5(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes()).hexdigest()


def _jit_key(model, free_idx, lai_mode, obs):
    """Everything that affects the traced graph (not the data/init)."""
    return (
        int(model.tree_type), int(model.lad_type), float(model.diam),
        float(model.ro_soil), float(model.ro_L), float(model.tau_L),
        int(model.nz), float(model.hotspot), bool(model.full_flat),
        float(model.slope), float(model.aspect),
        None if model.alpha_slope is None else float(model.alpha_slope),
        tuple(free_idx), bool(lai_mode),
        float(obs.sza), float(obs.saa),
        _arr_hash(obs.vza), _arr_hash(obs.vaa),
    )


def run_inversion(
    obs: Observation,
    init_params,
    free_names=("H", "cover", "d_L", "slope", "aspect"),
    bounds=DEFAULT_BOUNDS,
    model: ModelConfig | None = None,
    n_iter: int = 300,
    lr: float = 0.02,
    prior_weight: float = 1e-3,
    verbose=False,
    progress_cb=None,
    parameterization: str = "d_L",
    prior_sigma=None,
    prior_mean=None,
):
    """Invert selected parameters from multi-angle BRF.

    ``parameterization="LAI"`` interprets the d_L slot as LAI and converts to
    d_L = LAI / (H * Lambda(cover)) at every evaluation.  This is the
    identifiable parameterization for BRF-only inversions: the SRT BRF is
    exactly invariant under H -> aH, d_L -> d_L/a (LAI and cover fixed).

    ``prior_sigma`` (5-vector, physical/effective units) adds a Gaussian prior
    ``0.5 sum(((x - prior_mean)/prior_sigma)^2)``; use ``np.inf`` for
    uninformative components.  H has zero BRF sensitivity once LAI/cover are
    fixed, so its posterior then equals this prior.

    Compiled ``value_and_grad``/Jacobian executables are cached by model
    configuration and observation geometry; init, bounds, data and sigma are
    function arguments, so repeated runs do not recompile.
    """
    model = model or ModelConfig()
    free_names = list(free_names)
    lai_mode = parameterization == "LAI" or "LAI" in free_names
    if lai_mode:
        free_names = ["d_L" if n == "LAI" else n for n in free_names]
    bounds = np.asarray(bounds, dtype=float)
    init_np = np.asarray(init_params, dtype=float)
    lo_np = bounds[:, 0]
    hi_np = bounds[:, 1]
    init = jnp.asarray(init_np)
    lo = jnp.asarray(lo_np)
    hi = jnp.asarray(hi_np)
    free_idx = tuple(PARAM_NAMES.index(name) for name in free_names)
    fidx = jnp.asarray(free_idx)
    z0 = encode(init[fidx], lo[fidx], hi[fidx])

    sigma = float(obs.sigma)
    obs_brf = jnp.asarray(obs.brf, dtype=float)
    use_gauss_prior = prior_sigma is not None

    # Unify ridge and Gaussian priors: ridge == Gaussian(init, 1/sqrt(2w)).
    if use_gauss_prior:
        ps_np = np.asarray(prior_sigma, dtype=float)
        pm_np = np.asarray(prior_mean if prior_mean is not None else init_np, dtype=float)
    else:
        pm_np = init_np.copy()
        if prior_weight and prior_weight > 0.0:
            ps_np = np.full(5, 1.0 / np.sqrt(2.0 * float(prior_weight)))
        else:
            ps_np = np.full(5, np.inf)
    pm = jnp.asarray(pm_np)
    ps = jnp.asarray(ps_np)

    def to_phys(z, init_, lo_, hi_):
        x = _full_params(z, init_, free_idx, lo_, hi_)
        if lai_mode:
            x = x.at[2].set(dl_from_lai(x[2], x[0], x[1], model.tree_type))
        return x

    def loss_fn(z, init_, lo_, hi_, pm_, ps_, sigma_, target_):
        x = to_phys(z, init_, lo_, hi_)
        pred = predict(x, obs, model)
        mse = jnp.mean(((pred - target_) / sigma_) ** 2)
        x_eff = (x.at[2].set(lai_from_dl(x[2], x[0], x[1], model.tree_type))
                 if lai_mode else x)
        prior = 0.5 * jnp.sum(((x_eff - pm_) / ps_) ** 2)
        return mse + prior

    def pred_free(z, init_, lo_, hi_):
        return predict(to_phys(z, init_, lo_, hi_), obs, model)

    key = _jit_key(model, free_idx, lai_mode, obs)
    loss_grad = _cache(key + ("lg",),
                       lambda: jax.jit(jax.value_and_grad(loss_fn, argnums=0)))
    jac_fn = _cache(key + ("jac",),
                    lambda: jax.jit(jax.jacrev(pred_free, argnums=0)))
    pred_jit = _cache(key + ("pred",),
                      lambda: jax.jit(lambda x: predict(x, obs, model)))

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(z0)

    history = []
    z = z0
    for i in range(n_iter):
        loss, grads = loss_grad(z, init, lo, hi, pm, ps, sigma, obs_brf)
        updates, opt_state = optimizer.update(grads, opt_state, z)
        z = optax.apply_updates(z, updates)
        history.append(float(loss))
        if verbose and (i % 25 == 0 or i == n_iter - 1):
            print(f"iter {i:4d} loss {float(loss):.6f}", flush=True)
        if progress_cb is not None and (i % 10 == 0 or i == n_iter - 1):
            progress_cb(i, n_iter, float(loss))
        if i > 20 and abs(history[-1] - history[-2]) < 1e-10:
            break

    x_opt = to_phys(z, init, lo, hi)
    pred = pred_jit(x_opt)

    jac = jac_fn(z, init, lo, hi)
    jac_np = np.asarray(jac)
    fisher = jac_np.T @ jac_np / (sigma**2)
    if use_gauss_prior:
        # Transform to effective-parameter space: dx/dz is diagonal.
        u = np.asarray(jax.nn.sigmoid(z))
        dxdz = (hi_np - lo_np)[np.asarray(free_idx)] * u * (1.0 - u)
        Dinv = np.diag(1.0 / np.maximum(dxdz, 1e-12))
        F_x = Dinv @ fisher @ Dinv
        ps_np_free = ps_np[np.asarray(free_idx)]
        prec = np.where(np.isfinite(ps_np_free),
                        1.0 / np.maximum(ps_np_free, 1e-12) ** 2, 0.0)
        F_post = F_x + np.diag(prec)
        cov = np.linalg.pinv(F_post)
        fisher_out = F_post
    else:
        cov = np.linalg.pinv(fisher)
        fisher_out = fisher
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.maximum(np.outer(std, std), 1e-12)
    cond = float(np.linalg.cond(fisher_out)) if np.size(fisher_out) else float("inf")

    lai_opt = lai_from_dl(x_opt[2], x_opt[0], x_opt[1], model.tree_type)
    retrieved = {name: float(x_opt[i]) for i, name in enumerate(PARAM_NAMES)}
    retrieved["LAI"] = float(lai_opt)
    if lai_mode:
        free_retrieved = {
            ("LAI" if name == "d_L" else name):
            float(lai_opt if name == "d_L" else x_opt[PARAM_NAMES.index(name)])
            for name in free_names
        }
        out_names = ["LAI" if n == "d_L" else n for n in free_names]
    else:
        free_retrieved = {name: float(x_opt[PARAM_NAMES.index(name)]) for name in free_names}
        out_names = list(free_names)
    return {
        "free_names": out_names,
        "retrieved": retrieved,
        "free_retrieved": free_retrieved,
        "loss_history": history,
        "predicted": np.asarray(pred).tolist(),
        "jacobian": jac_np.tolist(),
        "fisher": np.asarray(fisher_out).tolist(),
        "covariance": cov.tolist(),
        "std": std.tolist(),
        "correlation": corr.tolist(),
        "condition_number": cond,
        "rmse": float(np.sqrt(np.mean((np.asarray(pred) - obs.brf) ** 2))),
    }


def run_sensitivity(
    params,
    obs: Observation,
    model: ModelConfig | None = None,
):
    """Jacobian / Fisher information at a given physical parameter point."""
    model = model or ModelConfig()
    x = jnp.asarray(params, dtype=float)
    sigma = float(obs.sigma)

    def pred_all(xv):
        return predict(xv, obs, model)

    key = _jit_key(model, (), False, obs)
    jac_fn = _cache(key + ("sens",), lambda: jax.jit(jax.jacrev(pred_all)))
    pred_jit = _cache(key + ("predall",), lambda: jax.jit(pred_all))
    jac = jac_fn(x)
    jac_np = np.asarray(jac)
    fisher = jac_np.T @ jac_np / (sigma**2)
    cov = np.linalg.pinv(fisher)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.maximum(np.outer(std, std), 1e-12)
    cond = float(np.linalg.cond(fisher))
    pred = np.asarray(pred_jit(x))
    # Relative sensitivity: dBRF/dtheta * theta / BRF.
    rel = jac_np * np.asarray(x)[None, :] / np.maximum(pred[:, None], 1e-9)
    return {
        "params": [float(v) for v in x],
        "names": list(PARAM_NAMES),
        "predicted": pred.tolist(),
        "jacobian": jac_np.tolist(),
        "relative_sensitivity": rel.tolist(),
        "fisher": fisher.tolist(),
        "covariance": cov.tolist(),
        "std": std.tolist(),
        "correlation": corr.tolist(),
        "condition_number": cond,
    }
