# FLOWFS

Simulation, calibration, and phase retrieval for FLOWFS on MagAO-X, built on
HCIPy. The code models wavefront propagation, fits aberrations from recorded
images, and compares linear and nonlinear wavefront reconstruction.

Requires Python 3.12 or newer. Install with [uv](https://docs.astral.sh/uv/)
from the repository root:

```bash
uv sync --python 3.12
```

Dependencies are specified in `pyproject.toml`. 

The main entry points are:

- `flowfs/`: the `FLOWFS` model, phase retrieval, data loading, and plotting.
- `examples/calibration_from_sweep.py`: calibrate from recorded mode sweeps.
- `examples/injection_retrieval.py`: reconstruct recorded aberration injections.
- `scripts/`: batch reduction commands for specific datasets.
- `tests/`: configuration, gradient, and convergence checks.

Run scripts from the repository root so relative paths resolve correctly.
For example, replace the input path with a recorded sweep in the format expected
by the calibration script, and select a matching configuration:

```bash
uv run python examples/calibration_from_sweep.py path/to/sweep.pkl \
    --from-config configs/flowfs_config_lyot_sm.json --num-workers 4
```

Use `--help` to see script options. Reduction workflows write calibrated
configurations to `configs/`, arrays to `results/`, and figures to `plots/`.
Recorded data and generated arrays and figures are not included in version
control. The batch scripts contain dataset-specific paths that need adapting.
