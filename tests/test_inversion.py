"""Synthetic inversion and sensitivity checks for the JAX model."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from diff_srt.inversion import (
    ModelConfig,
    Observation,
    predict,
    run_inversion,
    run_sensitivity,
)


def _synthetic():
    vza = np.array([0.0, 15.0, 30.0, 45.0, 60.0])
    vaa = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    model = ModelConfig(nz=40)
    true = np.array([5.2, 0.52, 1.1, 15.0, 270.0])
    obs = Observation(
        sza=45.0, saa=270.0, vza=vza, vaa=vaa,
        brf=np.zeros(vza.size * vaa.size), sigma=0.003,
    )
    clean = np.asarray(predict(jax.numpy.asarray(true), obs, model))
    rng = np.random.default_rng(1)
    obs.brf = clean * (1.0 + 0.005 * rng.standard_normal(clean.shape))
    return model, true, obs


def test_synthetic_inversion_recovers_structure():
    model, true, obs = _synthetic()
    init = np.array([4.5, 0.45, 1.1, 8.0, 270.0])
    out = run_inversion(
        obs,
        init,
        free_names=("H", "cover", "slope"),
        model=model,
        n_iter=250,
        lr=0.01,
    )
    assert out["rmse"] < 0.004
    assert abs(out["free_retrieved"]["H"] - true[0]) / true[0] < 0.15
    assert abs(out["free_retrieved"]["cover"] - true[1]) / true[1] < 0.15
    assert abs(out["free_retrieved"]["slope"] - true[3]) / true[3] < 0.20


def test_sensitivity_shapes_and_no_nan():
    model, true, obs = _synthetic()
    out = run_sensitivity(jax.numpy.asarray(true), obs, model)
    assert np.asarray(out["jacobian"]).shape == (obs.brf.size, 5)
    assert np.asarray(out["relative_sensitivity"]).shape == (obs.brf.size, 5)
    assert np.all(np.isfinite(np.asarray(out["jacobian"])))
    # H and d_L are structurally collinear; the Fisher matrix is singular.
    assert np.asarray(out["correlation"]).shape == (5, 5)
