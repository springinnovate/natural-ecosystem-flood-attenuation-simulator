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

The initial entry point validates the configuration shape, creates the configured output directory, logs setup progress, and exits without running a simulation.
