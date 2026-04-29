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

The setup step writes geospatial intermediates beneath the configured output directory, using the configuration file name as the workspace name. For example, `configs/united_states.yaml` writes to `outputs/united_states/`.
