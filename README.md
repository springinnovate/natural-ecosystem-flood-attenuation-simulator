# NEFAS - Natural Ecosystem Flood Attenuation Simulator

NEFAS is a reduced-complexity 2D flood simulation model for evaluating how wetlands, floodplains, forests, and other natural landscapes slow, store, and attenuate floodwaters.

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
