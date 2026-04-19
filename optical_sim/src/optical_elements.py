"""
optical_elements.py
Dataclass definitions for source, lens, and sensor.
These are pure data containers — no physics logic here.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class GaussianSource:
    """Laser source modelled as a fundamental Gaussian beam (TEM00)."""
    wavelength: float          # [m]
    beam_waist: float          # 1/e² intensity radius at waist [m]
    power: float = 1.0         # normalised power [W]
    source_type: str = "gaussian"

    @property
    def rayleigh_range(self) -> float:
        """Rayleigh range z_R = π w₀² / λ"""
        return np.pi * self.beam_waist ** 2 / self.wavelength

    @property
    def wavenumber(self) -> float:
        return 2.0 * np.pi / self.wavelength


@dataclass
class ThinLens:
    """Ideal thin-lens element."""
    focal_length: float        # [m]
    clear_aperture: float      # usable diameter [m]
    diameter: float = 0.0254   # physical diameter [m]


@dataclass
class Sensor:
    """Camera / sensor plane."""
    resolution: Tuple[int, int] = (512, 512)   # (H, W) in pixels
    pixel_pitch: float = 5.5e-6                 # [m/pixel]

    @property
    def sensor_size(self) -> Tuple[float, float]:
        """Physical sensor size (H, W) in metres."""
        return (self.resolution[0] * self.pixel_pitch,
                self.resolution[1] * self.pixel_pitch)


@dataclass
class Alignment:
    """Misalignment / positioning parameters of the sensor."""
    x_offset: float = 0.0     # lateral [m]
    y_offset: float = 0.0     # lateral [m]
    tilt_x: float = 0.0       # [rad]
    tilt_y: float = 0.0       # [rad]
    defocus: float = 0.0      # extra axial shift [m]


@dataclass
class OpticalSetup:
    """Complete laser → lens → camera configuration."""
    source: GaussianSource
    lens: ThinLens
    sensor: Sensor
    alignment: Alignment
    laser_to_lens: float       # [m]
    lens_to_camera: float      # [m]

    # simulation grid params
    grid_size: int = 1024
    grid_extent: float = 0.03  # half-width [m]
    propagation_backend: str = "fresnel_numpy"

    @property
    def effective_camera_distance(self) -> float:
        """lens_to_camera + defocus"""
        return self.lens_to_camera + self.alignment.defocus


# --------------- factory from flat dict / YAML ---------------

def setup_from_dict(cfg: dict) -> OpticalSetup:
    """Build an OpticalSetup from a nested config dict (matches base_config.yaml)."""
    src = cfg["source"]
    ln = cfg["lens"]
    sn = cfg["sensor"]
    geo = cfg["geometry"]
    ali = cfg.get("alignment", {})
    sim = cfg.get("simulation", {})

    return OpticalSetup(
        source=GaussianSource(
            wavelength=src["wavelength"],
            beam_waist=src["beam_waist"],
            power=src.get("power", 1.0),
            source_type=src.get("type", "gaussian"),
        ),
        lens=ThinLens(
            focal_length=ln["focal_length"],
            clear_aperture=ln["clear_aperture"],
            diameter=ln.get("diameter", 0.0254),
        ),
        sensor=Sensor(
            resolution=tuple(sn["resolution"]),
            pixel_pitch=sn["pixel_pitch"],
        ),
        alignment=Alignment(
            x_offset=ali.get("x_offset", 0.0),
            y_offset=ali.get("y_offset", 0.0),
            tilt_x=ali.get("tilt_x", 0.0),
            tilt_y=ali.get("tilt_y", 0.0),
            defocus=ali.get("defocus", 0.0),
        ),
        laser_to_lens=geo["laser_to_lens"],
        lens_to_camera=geo["lens_to_camera"],
        grid_size=sim.get("grid_size", 1024),
        grid_extent=sim.get("grid_extent", 0.03),
        propagation_backend=sim.get("propagation_backend", "fresnel_numpy"),
    )
