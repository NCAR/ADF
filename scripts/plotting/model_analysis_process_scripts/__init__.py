"""
Model analysis modules.

This package contains individual analysis scripts for COMPASS nudged
model diagnostics. Each module can be imported and called
independently or orchestrated through the model analysis wrappers.
"""

from .sensitivity_vs_sigmaw_2x2_obs_cam import sensitivity_vs_sigmaw_2x2_obs_cam
from .alpha_sigmaw_regime_comparison_2x2 import alpha_sigmaw_regime_comparison_2x2

__all__ = [
    'sensitivity_vs_sigmaw_2x2_obs_cam',
    'alpha_sigmaw_regime_comparison_2x2',
]
