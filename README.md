# GVV Volatility Surface

An implementation of the Gamma-Vanna-Volga (GVV) model for constructing volatility surfaces. Includes two calibration methods: a least squares optimization with SciPy and a basic linear algebra method.
Also includes a notebook that demonstrates the calibration of the three model parameters (σ, η, and ρ), construction of the implied volatility surface, and a brief example of how theta can be decomposed into contributions from gamma, vanna, and volga.
