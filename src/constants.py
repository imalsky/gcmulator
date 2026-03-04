from __future__ import annotations

import math

# Time conversion constants.
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0

# Shared rollout constraints.
MIN_ROLLOUT_STEPS = 1

# Numerical stability floors/clips.
STD_FLOOR = 1.0e-12
LOG10_INVERSE_CLIP_MIN = -30.0
LOG10_INVERSE_CLIP_MAX = 30.0
PARAM_NORM_CLIP_ABS = 1.0e6

# Sampling and probability constraints.
PROBABILITY_MIN = 0.0
PROBABILITY_MAX = 1.0

# Coordinate and basis generation constants.
TWO_PI = 2.0 * math.pi
RANDOM_BASIS_AMP_MIN = 0.25
RANDOM_BASIS_AMP_MAX = 1.0

# Canonical MY_SWAMP rest-state initialization.
CANONICAL_VORTICITY_FACTOR = 2.0
