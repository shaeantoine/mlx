from abc import ABC, abstractmethod
from typing import Optional
import math

class Distribution(ABC): 
    def __init__(self, validate_args: bool = False): 
        self._validate_args = validate_args
    
    @property
    @abstractmethod
    def mean(self) -> float:
        "Return the mean of the distribution"
        pass
    
    @property
    @abstractmethod
    def variance(self) -> float:
        "Returns the variance of the distribution"
        pass
    
    @abstractmethod
    def sample(self) -> float:
        "Draw a random sample from the distribution"
        pass

    @abstractmethod
    def log_prob(self, value: float) -> float: 
        "Compute the logarithm of a probability of the given value"
        pass

    def prob(self, value: float) -> float:
        "Return the probability density of a given value"
        return math.exp(self.log_prob(value))
    
    def entropy(self) -> Optional[float]: 
        raise NotImplementedError("Entropy is not implemented for this distribution")
    
