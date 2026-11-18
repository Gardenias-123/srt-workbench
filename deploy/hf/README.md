---
title: SRT Workbench
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SRT Workbench

Differentiable stochastic radiative transfer (SRT / TopoSRT) canopy model with
forward simulation, gradient-based inversion and sensitivity analysis.

- JAX forward model validated against the Fortran reference to ~1e-5 in BRF
- Spherical / cylindrical / conical / ellipsoidal crowns, LAD 1–9, flat/sloping terrain
- Inversion of LAI and fractional cover (plus optional H prior), Fisher/identifiability analysis

Source code and documentation: see the GitHub repository linked in the Space card.

> This Space runs the JAX engine only. The Fortran reference binary is not
> distributed here.
