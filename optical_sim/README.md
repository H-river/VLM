# optical_sim — Gaussian Beam Propagation Simulator

A pure-Python physics engine that models a **laser → thin lens → camera sensor** optical system using Fresnel / angular-spectrum propagation.

## Overview

```
Laser source (TEM₀₀ Gaussian)
        │
        ▼  propagate (laser_to_lens)
    Thin lens + circular aperture
        │
        ▼  propagate (lens_to_camera + defocus)
    Sensor plane  ← alignment offsets (x, y, tilt)
        │
        ▼
    Intensity image (H × W)
```

The simulator generates realistic 2-D beam intensity profiles on a virtual camera sensor, with configurable misalignment (lateral offset, tilt, defocus). It is used to produce training data for the sibling [`lang2setup`](../lang2setup/) pipeline.

## Features

| Feature | Details |
|---|---|
| **Propagation backends** | Fresnel (transfer-function FFT), Angular Spectrum, waveprop placeholder |
| **Optical elements** | `GaussianSource`, `ThinLens`, `Sensor`, `Alignment` dataclasses |
| **Beam metrics** | Centroid, beam width (D4σ), ellipticity, peak intensity, encircled energy |
| **Dataset generation** | Random parameter sweeps with metadata + intensity output saved per sample |
| **Config-driven** | YAML configs for base setup, random sweeps, and parameter ranges |

## Project Structure

```
optical_sim/
├── __init__.py
├── configs/
│   ├── base_config.yaml      # Default optical setup (He-Ne, f=100mm, 1024×1024)
│   ├── random_config.yaml    # Random sweep parameter ranges
│   └── sweep_config.yaml     # Deterministic sweep grid
├── src/
│   ├── optical_elements.py   # Dataclasses: GaussianSource, ThinLens, Sensor, Alignment, OpticalSetup
│   ├── simulator.py          # Propagation engine: source field → lens → sensor
│   ├── metrics.py            # Beam profile analysis (centroid, width, ellipticity)
│   ├── experiment_generator.py  # Random/sweep config generation
│   ├── main_generate_dataset.py # CLI entry point for batch dataset generation
│   ├── io_utils.py           # File I/O helpers
│   └── __main__.py           # python -m src entry point
├── outputs/                  # Generated simulation data
│   └── random_5k/            # 5002 random samples
│       ├── summary.csv
│       ├── summary.jsonl
│       └── rand_XXXXX/       # Per-sample: metadata.json + intensity.npy
├── environment.yaml          # Conda environment spec
└── requirements.txt          # Pip requirements
```

## Quick Start

### Installation

```bash
# Option A: Conda (recommended)
conda env create -f optical_sim/environment.yaml
conda activate optical_sim

# Option B: Pip
pip install -r optical_sim/requirements.txt
```

### Run a single simulation

```python
import yaml
from optical_sim.src.optical_elements import setup_from_dict
from optical_sim.src.simulator import run_simulation

with open("optical_sim/configs/base_config.yaml") as f:
    cfg = yaml.safe_load(f)

# Set alignment offsets
cfg["alignment"]["x_offset"] = 0.001   # 1 mm lateral shift
cfg["alignment"]["tilt_x"] = 0.01      # 10 mrad tilt

setup = setup_from_dict(cfg)
result = run_simulation(setup)

# result["intensity"] is a 1024×1024 numpy array
```

### Generate a random dataset

```bash
cd VLM/
python -m optical_sim.src.main_generate_dataset
```

This produces `outputs/random_5k/` with per-sample `metadata.json` and `intensity.npy` files.

## Default Configuration

| Parameter | Value |
|---|---|
| Wavelength | 632.8 nm (He-Ne) |
| Beam waist (w₀) | 1.0 mm |
| Focal length | 100 mm |
| Clear aperture | 25 mm |
| Sensor | 1024 × 1024, 5.5 µm pitch |
| Laser → lens | 200 mm |
| Lens → camera | 150 mm |
| Grid | 1024 × 1024, ±30 mm extent |

## Physics

The propagation uses the **Transfer Function** form of the Fresnel integral:

$$U_{out}(x,y) = \mathcal{F}^{-1}\!\Big\{\mathcal{F}\{U_{in}\} \cdot H(f_x, f_y)\Big\}$$

where $H(f_x,f_y) = e^{jkz} \cdot e^{-j\pi\lambda z(f_x^2+f_y^2)}$

The thin lens applies phase $\exp\!\big(-\frac{jk}{2f}(x^2+y^2)\big)$ with a circular hard aperture.

## License

Research use only.
