# Copyright © 2023 Apple Inc.

import mlx.core as mx
from typing import Union, Optional, Any, Callable, Dict, List, Tuple, Type, get_type_hints
from typing_extensions import Annotated
import math
import inspect
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


@dataclass(frozen=True)
class CheckResult: 
    ok: mx.array                   # shape () boolean
    violations: mx.array | None    # boolean mask (same shape as value) or None
    summary: Dict[str, mx.array]   # small scalars/arrays

class Constraint: 
    def check(self, value: mx.array) -> CheckResult:
        raise NotImplementedError
    
    def describe(self) -> str:
        raise NotImplementedError
    
    def error_message(self, param_name: str) -> str: 
        raise NotImplementedError
    
    def __call__(self, value: mx.array) -> CheckResult: 
        return self.check(value)
    
    def __str__(self):
        return self.describe()
    

class DistributionValidationError(ValueError):
    """Enhanced exception for distribution parameter validation errors."""
    
    def __init__(self, message: str, param_name: str = None, actual_value: Any = None, 
                 constraint: 'Constraint' = None, suggestions: List[str] = None):
        super().__init__(message)
        self.param_name = param_name
        self.actual_value = actual_value
        self.constraint = constraint
        self.suggestions = suggestions or []
    
    def debug(self):
        """Print detailed debugging information."""
        print(f"Validation Error: {self}")
        if self.param_name:
            print(f"   Parameter: {self.param_name}")
        if self.actual_value is not None:
            print(f"   Actual value: {self.actual_value}")
        if self.constraint:
            print(f"   Required constraint: {self.constraint}")
        if self.suggestions:
            print("Suggestions:")
            for suggestion in self.suggestions:
                print(f"   • {suggestion}")
    
@dataclass(frozen=True)
class Scalar(Constraint):
    def check(self, value: mx.array) -> CheckResult: 
        ok = mx.array(value.ndim == 0)
        return CheckResult(ok=ok, violations=None, summary={"ndim": mx.array(value.ndim)})
    
    def describe(self) -> str:
        return "scalar"
    
    def error_message(self, param_name: str) -> str:
        return f"Parameter '{param_name}' must be scalar"

@dataclass(frozen=True)
class Positive(Constraint):
    min_value: float = 0.0
    def check(self, value: mx.array) -> CheckResult:
        mask = value > self.min_value
        ok = mx.all(mask)
        summary = {"min": mx.min(value), "num_bad": mx.size(value) - mx.sum(mask)}
        return CheckResult(ok=ok, violations=~mask, summary=summary)
    
    def describe(self) -> str:
        return "positive ( > 0.0)" 
    
    def error_message(self, param_name: str) -> str:
        return f"Parameter '{param_name}' must be {self.describe()}"

@dataclass(frozen=True)
class Interval(Constraint):
    lower: float = -math.inf
    upper: float = math.inf
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def __post_init__(self):
        if not (self.lower < self.upper):
            raise ValueError(f"Invalid interval: lower ({self.lower}) must be < upper ({self.upper})")

    def check(self, value: mx.array) -> CheckResult:
        low = (value >= self.lower) if self.lower_inclusive else (value > self.lower)
        high = (value <= self.upper) if self.upper_inclusive else (value < self.upper)
        mask = low & high
        ok = mx.all(mask)
        num_bad = mx.size(value) - mx.sum(mask)
        return CheckResult(ok=ok, violations=~mask, summary={"num_bad": num_bad})

    def describe(self) -> str:
        lb, ub = ("[", "]") if self.lower_inclusive else ("(", ")")
        if not self.upper_inclusive:
            ub = ")"
        return f"in {lb}{self.lower}, {self.upper}{ub}"

    def error_message(self, param_name: str) -> str:
        return f"Parameter '{param_name}' must be {self.describe()}"

@dataclass(frozen=True)
class Integer(Constraint):
    def check(self, value: mx.array) -> CheckResult:
        is_int_dtype = value.dtype in (mx.int8, mx.int16, mx.int32, mx.int64, getattr(mx, 'uint32', mx.int32))
        if is_int_dtype:
            ok = mx.array(True)
            return CheckResult(ok=ok, violations=None, summary={})

    def describe(self) -> str:
        return "integer values"
    
    def error_message(self, param_name: str) -> str:
        return f"Parameter '{param_name}' must be integer-valued"

# Predefined constraints
positive = Positive()
strictly_positive = Positive(min_value=1e-8)
non_negative = Interval(0.0, math.inf, True, True)
real = Real()
scalar = Scalar()
unit_interval = Interval(0.0, 1.0, True, True)
probability = unit_interval


@dataclass
class ValidationResult:
    value: Any
    warnings: List[str] = field(default_factory=list)


class DistributionValidationError(ValueError):
    def __init__(self, message: str, param_name: str | None = None,
                 constraint: 'Constraint' | None = None):
        super().__init__(message)
        self.param_name = param_name
        self.constaint = constraint
    
    def __post_init__(self):
        if self.warnings:
            for warning in self.warnings:
                warnings.warn(warning, UserWarning)


class Constraint(ABC):
    """Enhanced base class for parameter constraints."""
    
    @abstractmethod
    def check(self, value: mx.array) -> bool:
        """Check if value satisfies the constraint."""
        pass
    
    @abstractmethod
    def error_message(self, param_name: str, actual_value: Any = None) -> str:
        """Generate contextual error message."""
        pass
    
    def suggest_fix(self, param_name: str, actual_value: Any = None) -> List[str]:
        """Suggest fixes for constraint violations."""
        return []
    
    def debug_info(self, value: mx.array) -> Dict[str, Any]:
        """Provide debugging information."""
        return {"constraint": str(self), "satisfied": self.check(value)}
    
    def __call__(self, value: mx.array) -> bool:
        return self.check(value)


class PositiveConstraint(Constraint):
    """Constraint requiring all values to be positive (> 0)."""
    
    def __init__(self, min_value: float = 1e-6):
        self.min_value = min_value
    
    def check(self, value: mx.array) -> bool:
        return mx.all(value > self.min_value).item()
    
    def error_message(self, param_name: str, actual_value: Any = None) -> str:
        if actual_value is not None:
            return f"Parameter '{param_name}' must be positive > {self.min_value} (got {actual_value})"
        return f"Parameter '{param_name}' must be positive > {self.min_value}"
    
    def suggest_fix(self, param_name: str, actual_value: Any = None) -> List[str]:
        suggestions = [f"Use {param_name}={max(self.min_value * 10, 1.0)}"]
        if actual_value is not None and hasattr(actual_value, 'item'):
            val = actual_value.item() if hasattr(actual_value, 'item') else actual_value
            if val <= 0:
                suggestions.append(f"Replace negative/zero value with {abs(val) + self.min_value}")
        return suggestions
    
    def debug_info(self, value: mx.array) -> Dict[str, Any]:
        info = super().debug_info(value)
        info.update({
            "min_value": float(mx.min(value).item()),
            "num_violations": int(mx.sum(value <= self.min_value).item()),
            "threshold": self.min_value
        })
        return info
    
    def __str__(self):
        return f"positive(>{self.min_value})"


        # Per-parameter constraints
        for name, x in arrays.items():
            res_warnings: List[str] = []
            constraint = self._param_constraints.get(name)
            if not isinstance(x, mx.array):
                raise DistributionValidationError("Expected mx.array", param_name=name)
            
            constraint_res = constraint.check(x)
            if not bool(mx.asarray(constraint_res.ok).item()):
                raise DistributionValidationError(
                    constraint.error_message(name,
                                    self._summarize(x)),
                                    param_name=name,
                                    constraint=constraint)
            res_warnings.append(f"{name}: {constraint.describe()} ok")
            results[name] = ValidationResult(x, res_warnings)


class ScalarConstraint(Constraint):
    """Enhanced scalar constraint with shape debugging."""
    
    def check(self, value: mx.array) -> bool:
        return value.ndim == 0
    
    def error_message(self, param_name: str, actual_value: Any = None) -> str:
        if actual_value is not None and hasattr(actual_value, 'shape'):
            return f"Parameter '{param_name}' must be scalar (0-D), got shape {actual_value.shape}"
        return f"Parameter '{param_name}' must be scalar (0-dimensional)"
    
    def suggest_fix(self, param_name: str, actual_value: Any = None) -> List[str]:
        suggestions = [f"Use {param_name}=scalar_value instead of array"]
        if actual_value is not None and hasattr(actual_value, 'shape'):
            if actual_value.size == 1:
                suggestions.append(f"Extract single value: {param_name}={param_name}.item()")
            else:
                suggestions.append(f"Use single value from array or reshape to scalar")
        return suggestions
    
    def debug_info(self, value: mx.array) -> Dict[str, Any]:
        info = super().debug_info(value)
        info.update({
            "actual_shape": value.shape,
            "actual_ndim": value.ndim,
            "size": value.size
        })
        return info
    
    def __str__(self):
        return "scalar"

    @staticmethod
    def _ensure_array(value: Any, dtype: mx.Dtype | None = None) -> mx.array:
        if isinstance(value, mx.array):
            return value if dtype is None else mx.astype(value, dtype)
        if isinstance(value, bool):
            return mx.array(value, dtype or mx.bool_)
        if isinstance(value, int):
            return mx.array(value, dtype or mx.int32)
        if isinstance(value, float):
            return mx.array(value, dtype or mx.float32)
        return mx.array(value, dtype=dtype)


# Enhanced factory functions
def create_normal_validator(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> AdvancedDistributionValidator:
    """Create enhanced validator for Normal distribution."""
    validator = AdvancedDistributionValidator(validation_level)
    validator.register_constraint("loc", CompoundConstraint(real, scalar))
    validator.register_constraint("scale", CompoundConstraint(strictly_positive, scalar))
    
    # Add relationship constraint for numerical stability
    validator.register_relationship(
        ["loc", "scale"],
        RelationshipConstraint(
            lambda loc, scale: mx.all(scale > 0.001 * mx.abs(loc)).item(),
            "scale > 0.001 * |loc| (numerical stability)"
        )
    )
    
    # Add statistical validator for strict mode
    def validate_scale_reasonableness(scale_val):
        """Check if scale is in reasonable range for numerical stability."""
        scale = scale_val.item() if hasattr(scale_val, 'item') else float(scale_val)
        return {
            "valid": 1e-10 < scale < 1e10,
            "message": f"Scale {scale} may cause numerical issues"
        }
    
    validator.register_statistical_validator("scale", validate_scale_reasonableness)
    return validator


def create_uniform_validator(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> AdvancedDistributionValidator:
    """Create validator for Uniform distribution with low < high constraint."""
    validator = AdvancedDistributionValidator(validation_level)
    validator.register_constraint("low", CompoundConstraint(real, scalar))
    validator.register_constraint("high", CompoundConstraint(real, scalar))
    
    # Critical relationship constraint
    validator.register_relationship(
        ["low", "high"],
        RelationshipConstraint(
            lambda low, high: (high > low).item(),
            "high > low"
        )
    )
    
    return validator


# Example enhanced distribution class
class EnhancedNormalDistribution:
    """Normal distribution with advanced validation."""
    
    def __init__(self, loc: Union[float, mx.array] = 0.0, 
                 scale: Union[float, mx.array] = 1.0,
                 validation_level: ValidationLevel = ValidationLevel.STANDARD):
        
        self._validator = create_normal_validator(validation_level)
        
        # Validate parameters
        results = self._validator.validate_params(loc=loc, scale=scale)
        self.loc = results["loc"].value
        self.scale = results["scale"].value
        
        # Store validation info for debugging
        self._validation_info = {name: result.debug_info for name, result in results.items()}
    
    def debug_validation(self):
        """Print validation debugging information."""
        print("🔍 Validation Debug Info:")
        for param_name, info in self._validation_info.items():
            print(f"  {param_name}: {info}")
    
    def __repr__(self):
        doc = self._validator.generate_documentation()
        return f"Normal(loc={self.loc.item():.3f}, scale={self.scale.item():.3f})\n\n{doc}"


# Utility functions for gradual migration
def validate_with_suggestions(param_name: str, value: Any, constraint: Constraint) -> mx.array:
    """Standalone validation with helpful error messages."""
    try:
        value_array = mx.array(value, mx.float32)
        if not constraint.check(value_array):
            error = DistributionValidationError(
                constraint.error_message(param_name, value),
                param_name, value, constraint,
                constraint.suggest_fix(param_name, value)
            )
            error.debug()
            raise error
        return value_array
    except Exception as e:
        if isinstance(e, DistributionValidationError):
            raise
        raise DistributionValidationError(
            f"Validation failed for {param_name}: {e}",
            param_name, value, suggestions=["Check value type and format"]
        )
