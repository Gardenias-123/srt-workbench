"""JAX engine: slope-aware forward, gradient inversion, and sensitivity."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
# Persistent XLA compilation cache: survives process restarts when the
# directory is on a writable/persistent volume (e.g. a Space's storage).
_JAX_CACHE = os.environ.get("SRT_JAX_CACHE") or str(
    Path(tempfile.gettempdir()) / "srt_jax_cache"
)
try:
    Path(_JAX_CACHE).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _JAX_CACHE)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
except Exception:  # older jaxlib / read-only filesystem
    pass

import jax.numpy as jnp
import numpy as np

from diff_srt.forward import brf_topo
from diff_srt.inversion import (
    DEFAULT_BOUNDS,
    PARAM_NAMES,
    ModelConfig,
    Observation,
    lai_from_dl,
    predict,
    run_inversion,
    run_sensitivity,
)
from diff_srt.topography import slope_geometry
from ..params import SRTParams


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _can_use_full(p: SRTParams, free=()) -> bool:
    """The validated full solver covers spherical crowns, flat or sloping.

    Slope/aspect enter through host-side (NumPy) geometry transforms, so they
    must be fixed rather than inverted with this model.
    """
    return (
        int(p.tree_type) in (1, 2, 3, 4)
        and not ({"slope", "aspect"} & set(free))
    )


def _model(p: SRTParams, free=()) -> ModelConfig:
    return ModelConfig(
        tree_type=p.tree_type,
        lad_type=p.lad,
        diam=p.tree_diam,
        ro_soil=p.ro_soil,
        ro_L=p.ro_L,
        tau_L=p.tau_L,
        nz=max(2, min(p.nz, 40)),
        hotspot=p.hot_spot,
        full_flat=_can_use_full(p, free),
        slope=float(p.slope),
        aspect=float(p.aspect),
        alpha_slope=float(p.alpha_slope),
    )


def _x(p: SRTParams) -> np.ndarray:
    return np.array([p.H, p.cover, p.d_L, p.slope, p.aspect], dtype=float)


def _angles(p: SRTParams):
    zens, azis = p.view_grid()
    out = []
    for z, a in zip(zens, azis):
        sza_s, saa_s, vza_s, vaa_s = slope_geometry(
            p.sza, p.saa, z, a, p.slope, p.aspect
        )
        out.append(
            {
                "SZA": float(p.sza),
                "SAA": float(p.saa),
                "SZA_T": float(sza_s),
                "SAA_T": float(saa_s),
                "VZA": float(z),
                "VAA": float(a),
                "VZA_T": float(vza_s),
                "VAA_T": float(vaa_s),
            }
        )
    return out


def _observation(p: SRTParams, sigma: float = 0.005) -> Observation:
    zens, azis = p.view_grid()
    return Observation(
        sza=float(p.sza),
        saa=float(p.saa),
        vza=np.asarray(zens, dtype=float),
        vaa=np.asarray(azis, dtype=float),
        brf=np.zeros(len(zens)),
        sigma=sigma,
    )


def _fortran_observation(manager, job_id: str, sigma: float = 0.005) -> Observation:
    result = manager.result(job_id)
    if not result:
        raise ValueError("找不到指定的 Fortran 正演结果")
    angles = result.get("angles", [])
    brf = result.get("columns", {}).get("brf_plane", [])
    rows = [(a, b) for a, b in zip(angles, brf) if a["VZA"] < 85]
    if not rows:
        raise ValueError("Fortran 结果没有可用的观测角")
    return Observation(
        sza=float(rows[0][0]["SZA"]),
        saa=float(rows[0][0]["SAA"]),
        vza=np.array([a["VZA"] for a, _ in rows], dtype=float),
        vaa=np.array([a["VAA"] for a, _ in rows], dtype=float),
        brf=np.array([b for _, b in rows], dtype=float),
        sigma=sigma,
    )


def _forward_pair(p: SRTParams, model: ModelConfig):
    """Return (brf_plane, brf_slope) for the selected JAX model."""
    zens, azis = p.view_grid()
    if model.full_flat:
        from diff_srt.jax_srt import brf_flat_sphere
        brf = np.asarray(brf_flat_sphere(
            p.H, p.cover, p.d_L, p.sza, p.saa, zens, azis,
            p.ro_L, p.tau_L, p.ro_soil,
            hotspot=p.hot_spot, nz=model.nz, direct_fraction=p.i0,
            slope=p.slope, aspect=p.aspect,
            tree_type=p.tree_type, diam=p.tree_diam, lad=p.lad,
            alpha_slope=p.alpha_slope,
        ))
        return brf, brf
    brf_plane, brf_slope = brf_topo(
        p.H,
        p.cover,
        p.d_L,
        p.tree_type,
        p.lad,
        p.tree_diam,
        jnp.deg2rad(p.alpha_slope),
        p.sza,
        p.saa,
        jnp.asarray(zens),
        jnp.asarray(azis),
        p.slope,
        p.aspect,
        p.ro_soil,
        p.ro_L,
        p.tau_L,
        nz=model.nz,
        hotspot=model.hotspot if model.full_flat else 0.0,
    )
    return brf_plane, brf_slope


def run_jax_forward(p: SRTParams, job, manager) -> dict:
    manager.progress(job, 10.0, "JAX forward")
    model = _model(p)
    brf_plane, brf_slope = _forward_pair(p, model)
    manager.progress(job, 80.0, "JAX forward done")
    result = _clean({
        "engine": "jax",
        "params": p.to_dict(),
        "angles": _angles(p),
        "columns": {
            "brf_plane": np.asarray(brf_plane).tolist(),
            "brf_slope": np.asarray(brf_slope).tolist(),
        },
        "model_note": (
            "JAX 完整可微 SRT（球/柱/锥/椭冠，LAD 1–9，平坦/坡面，与 Fortran 对齐 ~1e-5）"
            if model.full_flat else
            "JAX reduced 模型（直射 + 单次散射）；坡面/非球冠时使用"
        ),
    })
    out = Path(job.params["_result_dir"])
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    lines = ["SZA,SAA,VZA,VAA,BRF_plane,BRF_slope"]
    for a, bp, bs in zip(result["angles"], result["columns"]["brf_plane"], result["columns"]["brf_slope"]):
        lines.append(f"{a['SZA']},{a['SAA']},{a['VZA']},{a['VAA']},{bp:.8g},{bs:.8g}")
    (out / "jax_brf.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def run_inversion_job(p: SRTParams, payload: dict, job, manager) -> dict:
    free = payload.get("free") or ["H", "cover", "LAI"]
    free = [f for f in free if f in PARAM_NAMES or f == "LAI"]
    lai_mode = "LAI" in free
    if lai_mode:
        free = [f for f in free if f != "d_L"]
    model = _model(p, free)
    sigma = float(payload.get("sigma", 0.005))
    source = payload.get("source", "synthetic")
    if source == "fortran":
        job_id = payload.get("job_id")
        if not job_id:
            raise ValueError("Fortran 观测模式需要 job_id")
        obs = _fortran_observation(manager, job_id, sigma)
        truth = None
    else:
        obs = _observation(p, sigma)
        x_true = jnp.asarray(_x(p))
        clean = np.asarray(predict(x_true, obs, model))
        noise = float(payload.get("noise", 0.005))
        rng = np.random.default_rng(7)
        obs.brf = clean * (1.0 + noise * rng.standard_normal(clean.shape))
        truth = {name: float(v) for name, v in zip(PARAM_NAMES, _x(p))}
        truth["LAI"] = float(lai_from_dl(p.d_L, p.H, p.cover, p.tree_type))

    if not free:
        raise ValueError("至少选择一个反演参数")
    init = np.asarray(payload.get("init") or _x(p), dtype=float).copy()
    bounds = np.asarray(payload.get("bounds") or DEFAULT_BOUNDS, dtype=float).copy()
    if lai_mode:
        # Slot 2 is re-interpreted as LAI; start 20% below the current value.
        lai_now = float(lai_from_dl(p.d_L, p.H, p.cover, p.tree_type))
        init[2] = 0.8 * lai_now
        if payload.get("bounds") is None:
            bounds[2] = np.array([0.1, 6.0])

    def cb(i, n, loss):
        manager.progress(job, 5.0 + 90.0 * (i + 1) / max(n, 1), f"iteration {i + 1}/{n}, loss={loss:.4g}")

    manager.progress(job, 5.0, "JAX inversion")
    out = run_inversion(
        obs,
        init,
        free_names=tuple(free),
        bounds=bounds,
        model=model,
        n_iter=int(payload.get("n_iter", 300)),
        lr=float(payload.get("lr", 0.01)),
        prior_weight=float(payload.get("prior_weight", 1e-3)),
        progress_cb=cb,
        parameterization="LAI" if lai_mode else "d_L",
    )
    manager.progress(job, 96.0, "inversion done")
    result = _clean({
        "engine": "jax",
        "kind": "inversion",
        "source": source,
        "params": p.to_dict(),
        "truth": truth,
        "free_names": out["free_names"],
        "retrieved": out["retrieved"],
        "free_retrieved": out["free_retrieved"],
        "loss_history": out["loss_history"],
        "predicted": out["predicted"],
        "observed": np.asarray(obs.brf).tolist(),
        "jacobian": out["jacobian"],
        "fisher": out["fisher"],
        "covariance": out["covariance"],
        "std": out["std"],
        "correlation": out["correlation"],
        "condition_number": out["condition_number"],
        "rmse": out["rmse"],
        "model_note": (
            ("JAX 完整可微 SRT（球/柱/锥/椭冠，LAD 1–9，平坦/坡面，与 Fortran 对齐 ~1e-5）"
             + ("；LAI 参数化：d_L = LAI/(H·∫p dz)" if lai_mode else ""))
            if model.full_flat else
            "JAX reduced 模型（直射 + 单次散射）；坡面/非球冠时使用"
        ),
    })
    result = _clean(result)
    out_dir = Path(job.params["_result_dir"])
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def run_sensitivity_job(p: SRTParams, payload: dict, job, manager) -> dict:
    model = _model(p)
    sigma = float(payload.get("sigma", 0.005))
    source = payload.get("source", "synthetic")
    if source == "fortran":
        job_id = payload.get("job_id")
        if not job_id:
            raise ValueError("Fortran 观测模式需要 job_id")
        obs = _fortran_observation(manager, job_id, sigma)
    else:
        obs = _observation(p, sigma)
        obs.brf = np.asarray(predict(jnp.asarray(_x(p)), obs, model))

    manager.progress(job, 10.0, "JAX Jacobian")
    x = jnp.asarray(_x(p))
    out = run_sensitivity(x, obs, model)
    manager.progress(job, 95.0, "sensitivity done")
    result = _clean({
        "engine": "jax",
        "kind": "sensitivity",
        "source": source,
        "params": p.to_dict(),
        "names": out["names"],
        "predicted": out["predicted"],
        "jacobian": out["jacobian"],
        "relative_sensitivity": out["relative_sensitivity"],
        "fisher": out["fisher"],
        "covariance": out["covariance"],
        "std": out["std"],
        "correlation": out["correlation"],
        "condition_number": out["condition_number"],
        "model_note": (
            "JAX 完整可微 SRT（球/柱/锥/椭冠，LAD 1–9，平坦/坡面，与 Fortran 对齐 ~1e-5）"
            if model.full_flat else
            "JAX reduced 模型（直射 + 单次散射）；坡面/非球冠时使用"
        ),
    })
    result = _clean(result)
    out_dir = Path(job.params["_result_dir"])
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result
