# Synthetic Scenarios

This directory is reserved for generated synthetic flood-model scenarios used
during development. Generated GeoTIFF, GeoPackage, YAML, and output files are
ignored by git so the repository stays light.

Use the in-memory factories when you are working on solver behavior:

```python
from nefas.synthetic_cases import long_slope

case = long_slope(shape=(80, 240), cell_size=30, slope=0.0005)
grid = case.grid()

print(case.expected_behavior)
print(grid.elevation.shape)
```

Export a case only when you want to exercise the full CLI/geospatial pipeline:

```python
from pathlib import Path

from nefas.synthetic_cases import bowl_with_spillway

case = bowl_with_spillway(shape=(80, 160), cell_size=30)
export = case.export(
    Path("tests/fixtures/synthetic_scenarios/bowl_with_spillway"),
    time_step_seconds=5,
    total_runtime_seconds=90 * 60,
    snapshot_interval_minutes=15,
)

print(export.config)
```

Then run the exported configuration:

```powershell
python run_model.py tests/fixtures/synthetic_scenarios/bowl_with_spillway/config.yaml
```

The standard cases are:

- `flat_plain`
- `long_slope`
- `bowl_with_spillway`
- `ridge_with_gap`
- `incised_floodplain`
- `roughness_patch`
- `open_boundary_drainage`

You can also export every standard case from a Python prompt:

```python
from pathlib import Path

from nefas.synthetic_cases import all_cases

root = Path("tests/fixtures/synthetic_scenarios")
for case in all_cases():
    case.export(root / case.name)
```
