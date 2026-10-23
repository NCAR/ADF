"""
Model analysis modules.

This package contains individual analysis scripts for COMPASS nudged
model diagnostics. Each module can be imported and called
independently or orchestrated through the model analysis wrappers.
"""

from .cam_obs_dropsonde_comp import cam_obs_dropsonde_comp

__all__ = [
    'cam_obs_dropsonde_comp',
]
