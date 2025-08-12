import math
import numpy as np
from typing import Optional
import mlx.core as mx

from .distributions import Distribution

class Normal(Distribution):
    def __init__(self, loc: float, scale: float): 
        self._loc = mx.array(loc)
        self._scale = mx.array(scale)

    def mean(self) -> float:
        return self._loc
    
    def variance(self) -> float: 
        return self._scale
    
    def sample(self) -> float:
        return mx.random.normal(self._loc, self._scale)

    def log_prob(self, value: float) -> float: 
        scale = self._scale
        log_unnormalized = -0.5 * math.squared_difference(
            x / self._scale, self._loc / self._scale)
        log_normalization = tf.constant(
            0.5 * np.log(2. * np.pi), dtype=self.dtype) + tf.math.log(scale)
        return log_unnormalized - log_normalization
