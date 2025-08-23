import math
import mlx.core as mx

from .distributions import Distribution

class Normal(Distribution):
    def __init__(self, loc: float, scale: float): 
        self._loc = mx.array(loc)
        self._scale = mx.array(scale)

    @property
    def mean(self) -> float:
        return self._loc
    
    @property
    def mode(self) -> float: 
        return self._loc
    
    @property
    def stddev(self) -> float:
        return self._scale

    @property
    def variance(self) -> float: 
        return math.exp(self.stddev)
    
    # Implements MLX's reparameterized normal sampling method
    def sample(self) -> float:
        return mx.random.normal(self._loc, self._scale)
    
    def log_prob(self, value: float) -> float: 
        var = self._scale ** 2
        log_scale = math.log(self._scale)
        return -((value - self._loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))