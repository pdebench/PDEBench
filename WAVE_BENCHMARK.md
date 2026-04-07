# Wave Equation & Klein-Gordon Benchmark

**Contributed by:** Greg Partin ([@gpartin](https://github.com/gpartin))

New PDE benchmark tasks for PDEBench: the **wave equation** and **Klein-Gordon equation** in 1D and 2D with periodic boundary conditions.

## Equations

**Wave equation:**
$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$$

**Klein-Gordon equation:**
$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u - \chi^2 u$$

Both solved on periodic domain $[0, 1]^d$ with random Fourier initial conditions.

## Data Generation

```bash
# 1D wave equation, c=1.0, 1000 samples
python -m pdebench.data_gen.gen_wave sim.c=1.0 sim.chi=0.0

# Vary wave speed
python -m pdebench.data_gen.gen_wave sim.c=0.1
python -m pdebench.data_gen.gen_wave sim.c=0.4
python -m pdebench.data_gen.gen_wave sim.c=2.0

# Klein-Gordon with mass parameter
python -m pdebench.data_gen.gen_wave sim.c=1.0 sim.chi=0.5
python -m pdebench.data_gen.gen_wave sim.c=1.0 sim.chi=1.0
python -m pdebench.data_gen.gen_wave sim.c=1.0 sim.chi=2.0
python -m pdebench.data_gen.gen_wave sim.c=1.0 sim.chi=5.0

# 2D wave equation
python -m pdebench.data_gen.gen_wave sim.c=1.0 sim.ndim=2 sim.xdim=128
```

## Solver Details

- **Method**: Leapfrog (Verlet) finite-difference, second-order in space and time
- **Boundary conditions**: Periodic
- **CFL**: dt = 0.5 * dx / (c * sqrt(ndim))
- **Spatial resolution**: 1024 (1D), 128×128 (2D)
- **Temporal resolution**: 201 snapshots over t ∈ [0, 2]
- **Initial conditions**: Random superposition of 5 Fourier modes, normalized to unit max amplitude

## HDF5 Format

Output follows PDEBench Format 1A:
- `"tensor"`: shape `(n_samples, t, x)` for 1D, `(n_samples, t, x, y)` for 2D
- `"x-coordinate"`: spatial grid
- `"t-coordinate"`: temporal grid

## Baseline Results

### 1D Wave Equation — FNO vs UNet

Trained with autoregressive 41-step rollout, `initial_step=10`, spatial resolution 256 (downsampled 4×), temporal resolution 51 (downsampled 4×), 100 epochs.

| Wave Speed c | FNO nRMSE | UNet nRMSE | Solver nRMSE |
|:---:|:---:|:---:|:---:|
| 0.1 | 0.101 | 0.537 | 1.68 × 10⁻⁴ |
| 0.4 | 0.112 | 0.978 | 6.86 × 10⁻⁴ |
| 1.0 | 0.099 | 0.934 | 1.71 × 10⁻³ |
| 2.0 | 0.095 | 0.632 | 3.33 × 10⁻³ |

- **FNO** achieves ~10% nRMSE, stable across wave speeds (spectral convolutions match wave dynamics)
- **UNet** fails catastrophically (54–98% nRMSE) — spatial convolutions cannot capture global wave propagation
- **Solver** achieves near-machine-precision (O(c² Δt²) discretization error)

### Klein-Gordon Cross-Parameter Generalization

FNO trained on one χ value, tested on all four. This tests whether neural surrogates can extrapolate across the mass parameter.

| Train χ \ Test χ | 0.5 | 1.0 | 2.0 | 5.0 |
|:---:|:---:|:---:|:---:|:---:|
| **0.5** | **0.093** | 0.096 | 0.174 | 0.891 |
| **1.0** | 0.097 | **0.094** | 0.154 | 0.877 |
| **2.0** | 0.170 | 0.150 | **0.095** | 0.789 |
| **5.0** | 0.783 | 0.771 | 0.707 | **0.098** |

**Key findings:**
1. **Small χ changes (≤2×)**: 1.0–1.9× degradation (FNO extrapolates acceptably)
2. **Large χ changes (≥5×)**: 7–10× degradation (catastrophic failure)
3. **Physical interpretation**: High χ transitions solutions from propagating to evanescent regime — a qualitative phase change that FNO cannot bridge

This generalization matrix provides a systematic test of neural PDE solver robustness to parameter variation.

## Why These Benchmarks Are Useful

1. **Wave equation** is a fundamental hyperbolic PDE missing from the current PDEBench lineup (which focuses on diffusion, reaction-diffusion, advection, and Navier-Stokes)
2. **Klein-Gordon** adds a parametric family dimension — the mass parameter χ smoothly interpolates between wave-like and dispersive/evanescent regimes
3. **Cross-parameter generalization matrix** is a new benchmark format that tests a practical deployment scenario: can a model trained at one parameter value predict at another?
4. **UNet failure** on wave equations is a clear architectural diagnostic — useful for method developers

## Model Configurations

- FNO: `pdebench/models/config/args/config_wave.yaml`
- Klein-Gordon: `pdebench/models/config/args/config_klein_gordon.yaml`

### Optional: Physics-Inspired Optimizer

For wave-equation PDE benchmarks, the [LAdam optimizer](https://pypi.org/project/ladam/) (`pip install ladam`) provides Laplacian-coupled learning rates that match the PDE structure. In our 64-experiment validation, LAdam achieved the strongest gains on PINN and regression tasks.

```python
from ladam import LAdam, ChiAnnealScheduler
optimizer = LAdam(model.parameters(), lr=1e-3, c2=1e-4)
scheduler = ChiAnnealScheduler(optimizer, total_steps=10000)
```

## Files Added

```
pdebench/data_gen/gen_wave.py                      # Data generation script
pdebench/data_gen/src/sim_wave.py                   # Wave/KG simulator
pdebench/data_gen/configs/wave.yaml                 # Hydra config
pdebench/models/config/args/config_wave.yaml        # FNO/UNet training config
pdebench/models/config/args/config_klein_gordon.yaml # KG training config
```
