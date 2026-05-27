# NEFAS - Natural Ecosystem Flood Attenuation Simulator

NEFAS is a reduced-complexity 2D flood simulation model for evaluating how wetlands, floodplains, forests, and other natural landscapes slow, store, and attenuate floodwaters.

## Numerical Model Context

NEFAS is being built as a screening-scale raster flood model, not as an
engineering certification model. The relevant numerical family is therefore
reduced-complexity 2D inundation modeling: storage cells on a DEM, water-surface
slope as the driver, wet/dry fronts, simple roughness/friction, and enough time
dynamics to reason about arrival time, depth, duration, and drainage.

The current implementation is intentionally simpler than the published models
below. It is a scaffold for model development: rainfall is applied through the
simulation loop, snapshots are written over model time, and the placeholder
water movement step currently routes water downhill to lower neighboring water
surfaces. The next modeling decision is which physically motivated reduced
equation set should replace that placeholder.

### Candidate Solver Families

**Local-inertial shallow-water solvers.** Bates, Horritt, and Fewtrell (2010)
derive a simple inertial formulation for efficient 2D flood inundation models.
It keeps local acceleration, pressure-gradient forcing from water-surface slope,
and friction, while dropping the full nonlinear advective momentum terms. This is
close to the original NEFAS design idea: more head difference produces more
flow, but with finite wave propagation and damping rather than instant terrain
filling. de Almeida, Bates, Freer, and Souvignet (2012) then propose stability
improvements for that formulation, and de Almeida and Bates (2013) evaluate when
the local-inertial approximation agrees well with full shallow-water dynamics,
especially in lower subcritical flows.

**LISFLOOD-FP and subgrid-channel models.** LISFLOOD-FP is a major real-world
example of this reduced-complexity modeling lineage. Neal, Schumann, and Bates
(2012) extend LISFLOOD-FP with a subgrid channel model for large, data-sparse
domains where the DEM grid is coarser than the actual river channel. That is
important for regional river hydraulics, but it is probably not the first NEFAS
target while we are focused on DEM-based floodplain and wetland attenuation
without explicit channels. Sharifian et al. (2023) describe LISFLOOD-FP 8.1,
including GPU-accelerated local-inertial solvers and non-uniform grids.

**2D diffusive-wave solvers.** A diffusive-wave approximation drops inertial
terms and routes water from water-surface slope and friction. Leandro, Chen, and
Schumann (2014) present P-DWave, a parallel 2D diffusive-wave model with
variable time stepping for floodplain inundation. This family may be the best
near-term target for NEFAS because wetland and floodplain attenuation is often
slow, shallow, subcritical, and gradually varied. It is less appropriate for
dam-breaks, levee failures, steep flashy flows, or cases where momentum and wave
celerity dominate.

### Working Modeling Direction

The practical path is to keep the code architecture solver-neutral, but treat a
2D diffusive-wave solver as the first physical solver to study. That matches the
natural-ecosystem context: broad floodplain storage, ponding, slow release, and
terrain/roughness-controlled attenuation. A local-inertial solver remains a
strong follow-on option when we need better flood-wave timing, faster transients,
or comparison against the LISFLOOD-FP-style literature.

The current downhill-routing step should therefore be treated as a baseline and
debugging aid only. It is useful for exercising configuration, DEM clipping,
rainfall forcing, wet/dry behavior, open drainage, snapshots, and performance
instrumentation, but it should not be interpreted as a validated hydraulic
method.

### Feasibility Study

A useful feasibility study would compare three stages under the same raster
state model:

1. Current simple downhill routing as a nonphysical baseline.
2. A 2D diffusive-wave solver with Manning friction and adaptive or constrained
   timestepping.
3. A 2D local-inertial solver following the Bates/de Almeida/LISFLOOD-FP family.

The study should start with controlled synthetic cases before real landscapes:
rainfall on a plane, filling and draining a depression, drainage through open
boundaries, wet/dry-front stability, and a broad floodplain routing case. Key
metrics should include mass conservation, runtime per cell-step, stable timestep
range, maximum depth, arrival time, duration above threshold, water lost through
open boundaries, and sensitivity to DEM resolution and Manning roughness.

The decision criterion is not "which solver is most complete"; it is "which
solver is accurate enough for screening-scale natural flood attenuation while
remaining simple, explainable, and fast on available DEM and land-cover data."

### References

- Bates, P. D., Horritt, M. S., and Fewtrell, T. J. (2010). [A simple inertial formulation of the shallow water equations for efficient two dimensional flood inundation modelling](https://doi.org/10.1016/j.jhydrol.2010.03.027). Journal of Hydrology, 387(1-2), 33-45.
- de Almeida, G. A. M., Bates, P. D., Freer, J. E., and Souvignet, M. (2012). [Improving the stability of a simple formulation of the shallow water equations for 2-D flood modeling](https://doi.org/10.1029/2011WR011570). Water Resources Research, 48(5), W05528.
- de Almeida, G. A. M., and Bates, P. (2013). [Applicability of the local inertial approximation of the shallow water equations to flood modeling](https://doi.org/10.1002/wrcr.20366). Water Resources Research, 49(8), 4833-4844.
- Neal, J., Schumann, G., and Bates, P. (2012). [A subgrid channel model for simulating river hydraulics and floodplain inundation over large and data sparse areas](https://doi.org/10.1029/2012WR012514). Water Resources Research, 48(11), W11506.
- Sharifian, M. K., Kesserwani, G., Chowdhury, A. A., Neal, J., and Bates, P. (2023). [LISFLOOD-FP 8.1: new GPU-accelerated solvers for faster fluvial/pluvial flood simulations](https://doi.org/10.5194/gmd-16-2391-2023). Geoscientific Model Development, 16, 2391-2413.
- Leandro, J., Chen, A. S., and Schumann, A. (2014). [A 2D parallel diffusive wave model for floodplain inundation with variable time step (P-DWave)](https://doi.org/10.1016/j.jhydrol.2014.05.020). Journal of Hydrology, 517, 250-259.

## Run Setup

Install the project in editable mode:

```powershell
python -m pip install -e .
```

Prepare a model run from a YAML configuration:

```powershell
python run_model.py examples/minimal_config.yaml
```

Configure model timing in YAML with `simulation_time`:

```yaml
simulation_time:
  time_step_seconds: 5
  max_time_step_seconds: 30
  total_runtime_seconds: 172800
```

`total_runtime_seconds` controls the full simulation duration, so model runs can
continue after the rainfall series ends and capture drainage.

## Line Profiling

The main hydraulic timestep functions and snapshot renderer are decorated for
line-level profiling with `line_profiler`. Enable profiling for a run with:

```powershell
$env:LINE_PROFILE = "1"
python run_model.py examples/minimal_config.yaml
```

When the run exits, `line_profiler` prints per-line hit counts, total time, and
time per hit for the profiled functions.
