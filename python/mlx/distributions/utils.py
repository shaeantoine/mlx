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


class ValidationLevel(Enum):
    """Validation strictness levels."""
    OFF = 0          # No validation
    BASIC = 1        # Type checking only
    STANDARD = 2     # Standard constraints
    STRICT = 3       # All constraints + statistical tests
    DEBUG = 4        # Everything + detailed debugging info


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
    
    def suggest_fix(self):
        """Provide automatic fix suggestions."""
        if not self.suggestions:
            print("No automatic suggestions available")
        else:
            print("Try these fixes:")
            for i, suggestion in enumerate(self.suggestions, 1):
                print(f"{i}. {suggestion}")


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    value: mx.array
    warnings: List[str]
    debug_info: Dict[str, Any]
    
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


class IntervalConstraint(Constraint):
    """Enhanced interval constraint with better error handling."""
    
    def __init__(self, lower: float = -math.inf, upper: float = math.inf, 
                 lower_inclusive: bool = True, upper_inclusive: bool = True):
        self.lower = lower
        self.upper = upper
        self.lower_inclusive = lower_inclusive
        self.upper_inclusive = upper_inclusive
        
        if lower >= upper:
            raise ValueError(f"Invalid interval: lower ({lower}) >= upper ({upper})")
    
    def check(self, value: mx.array) -> bool:
        if self.lower_inclusive:
            lower_ok = mx.all(value >= self.lower).item()
        else:
            lower_ok = mx.all(value > self.lower).item()
        
        if self.upper_inclusive:
            upper_ok = mx.all(value <= self.upper).item()
        else:
            upper_ok = mx.all(value < self.upper).item()
        
        return lower_ok and upper_ok
    
    def error_message(self, param_name: str, actual_value: Any = None) -> str:
        lower_bracket = "[" if self.lower_inclusive else "("
        upper_bracket = "]" if self.upper_inclusive else ")"
        interval_str = f"{lower_bracket}{self.lower}, {self.upper}{upper_bracket}"
        
        if actual_value is not None:
            return f"Parameter '{param_name}' must be in interval {interval_str} (got {actual_value})"
        return f"Parameter '{param_name}' must be in interval {interval_str}"
    
    def suggest_fix(self, param_name: str, actual_value: Any = None) -> List[str]:
        suggestions = []
        mid_point = (self.lower + self.upper) / 2 if not math.isinf(self.lower) and not math.isinf(self.upper) else 0.5
        suggestions.append(f"Try {param_name}={mid_point}")
        
        if actual_value is not None:
            try:
                val = actual_value.item() if hasattr(actual_value, 'item') else float(actual_value)
                if val < self.lower:
                    suggestions.append(f"Increase value to at least {self.lower}")
                elif val > self.upper:
                    suggestions.append(f"Decrease value to at most {self.upper}")
            except:
                pass
        
        return suggestions
    
    def debug_info(self, value: mx.array) -> Dict[str, Any]:
        info = super().debug_info(value)
        info.update({
            "interval": f"{self.lower} to {self.upper}",
            "actual_min": float(mx.min(value).item()),
            "actual_max": float(mx.max(value).item()),
            "violations_lower": int(mx.sum(value < self.lower).item()) if not math.isinf(self.lower) else 0,
            "violations_upper": int(mx.sum(value > self.upper).item()) if not math.isinf(self.upper) else 0,
        })
        return info


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


class CompoundConstraint(Constraint):
    """Enhanced compound constraint with detailed error reporting."""
    
    def __init__(self, *constraints: Constraint):
        if not constraints:
            raise ValueError("CompoundConstraint requires at least one constraint")
        self.constraints = constraints
    
    def check(self, value: mx.array) -> bool:
        return all(constraint.check(value) for constraint in self.constraints)
    
    def error_message(self, param_name: str, actual_value: Any = None) -> str:
        failed_constraints = [
            constraint for constraint in self.constraints 
            if not constraint.check(value) if 'value' in locals() else True
        ]
        
        # If we have actual_value, find the first failing constraint properly
        if actual_value is not None:
            try:
                value = mx.array(actual_value) if not isinstance(actual_value, mx.array) else actual_value
                failed_constraints = [c for c in self.constraints if not c.check(value)]
            except:
                failed_constraints = self.constraints
        
        if failed_constraints:
            return failed_constraints[0].error_message(param_name, actual_value)
        return f"Parameter '{param_name}' violates compound constraint"
    
    def suggest_fix(self, param_name: str, actual_value: Any = None) -> List[str]:
        all_suggestions = []
        for constraint in self.constraints:
            all_suggestions.extend(constraint.suggest_fix(param_name, actual_value))
        return list(set(all_suggestions))  # Remove duplicates
    
    def debug_info(self, value: mx.array) -> Dict[str, Any]:
        info = {"constraint_type": "compound", "sub_constraints": {}}
        for i, constraint in enumerate(self.constraints):
            info["sub_constraints"][f"constraint_{i}"] = constraint.debug_info(value)
        return info
    
    def __str__(self):
        return " AND ".join(str(c) for c in self.constraints)


class RelationshipConstraint(Constraint):
    """Constraint for relationships between multiple parameters."""
    
    def __init__(self, relationship_fn: Callable, description: str, 
                 param_names: List[str] = None):
        self.relationship_fn = relationship_fn
        self.description = description
        self.param_names = param_names or []
    
    def check(self, *values) -> bool:
        """Check relationship between multiple values."""
        try:
            return self.relationship_fn(*values)
        except:
            return False
    
    def error_message(self, param_name: str, actual_value: Any = None) -> str:
        return f"Parameter relationship violated: {self.description}"
    
    def suggest_fix(self, param_name: str, actual_value: Any = None) -> List[str]:
        return [f"Ensure parameters satisfy: {self.description}"]
    
    def __str__(self):
        return f"relationship({self.description})"


# Pre-defined constraints with enhanced features
positive = PositiveConstraint()
strictly_positive = PositiveConstraint(min_value=1e-8)
non_negative = IntervalConstraint(0.0, math.inf, True, False)
real = IntervalConstraint(-math.inf, math.inf, True, True)
scalar = ScalarConstraint()
unit_interval = IntervalConstraint(0.0, 1.0, True, True)
probability = unit_interval


class AdvancedDistributionValidator:
    """Enhanced validator with multiple validation levels and debugging."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self._param_constraints: Dict[str, Constraint] = {}
        self._param_transforms: Dict[str, Callable] = {}
        self._relationship_constraints: List[Tuple[List[str], RelationshipConstraint]] = []
        self._statistical_validators: Dict[str, Callable] = {}
        
        # Performance optimization: pre-compile validation for common cases
        self._compiled_validators: Dict[str, Callable] = {}
    
    def set_validation_level(self, level: ValidationLevel):
        """Change validation level dynamically."""
        self.validation_level = level
        self._compiled_validators.clear()  # Clear cache when level changes
    
    def register_constraint(self, param_name: str, constraint: Constraint):
        """Register a constraint for a parameter."""
        self._param_constraints[param_name] = constraint
        # Clear compiled cache for this parameter
        if param_name in self._compiled_validators:
            del self._compiled_validators[param_name]
    
    def register_relationship(self, param_names: List[str], constraint: RelationshipConstraint):
        """Register a constraint between multiple parameters."""
        self._relationship_constraints.append((param_names, constraint))
    
    def register_statistical_validator(self, param_name: str, validator: Callable):
        """Register statistical validation (for STRICT level)."""
        self._statistical_validators[param_name] = validator
    
    def validate_param(self, param_name: str, value: Any, 
                      context: Dict[str, Any] = None) -> ValidationResult:
        """Enhanced parameter validation with detailed results."""
        warnings_list = []
        debug_info = {"param_name": param_name, "validation_level": self.validation_level.name}
        
        # Skip validation if OFF
        if self.validation_level == ValidationLevel.OFF:
            return ValidationResult(
                value=self._ensure_array(value),
                warnings=[],
                debug_info=debug_info
            )
        
        # Transform value
        if param_name in self._param_transforms:
            value = self._param_transforms[param_name](value)
        else:
            value = self._ensure_array(value)
        
        # Basic type checking
        if self.validation_level >= ValidationLevel.BASIC:
            if not isinstance(value, mx.array):
                raise DistributionValidationError(
                    f"Parameter '{param_name}' must be an MLX array",
                    param_name, value, suggestions=["Convert to mx.array first"]
                )
        
        # Standard constraint checking
        if self.validation_level >= ValidationLevel.STANDARD:
            if param_name in self._param_constraints:
                constraint = self._param_constraints[param_name]
                if not constraint.check(value):
                    raise DistributionValidationError(
                        constraint.error_message(param_name, value),
                        param_name, value, constraint,
                        constraint.suggest_fix(param_name, value)
                    )
                
                if self.validation_level >= ValidationLevel.DEBUG:
                    debug_info["constraint_debug"] = constraint.debug_info(value)
        
        # Strict statistical validation
        if self.validation_level >= ValidationLevel.STRICT:
            if param_name in self._statistical_validators:
                try:
                    stat_result = self._statistical_validators[param_name](value)
                    if not stat_result.get("valid", True):
                        warnings_list.append(f"Statistical validation warning for {param_name}: {stat_result.get('message', 'Unknown issue')}")
                except Exception as e:
                    warnings_list.append(f"Statistical validation failed for {param_name}: {e}")
        
        return ValidationResult(value, warnings_list, debug_info)
    
    def validate_params(self, **params) -> Dict[str, ValidationResult]:
        """Validate multiple parameters with relationship checking."""
        results = {}
        validated_values = {}
        
        # Validate individual parameters
        for name, value in params.items():
            result = self.validate_param(name, value, context=params)
            results[name] = result
            validated_values[name] = result.value
        
        # Check relationships (if STANDARD or higher)
        if self.validation_level >= ValidationLevel.STANDARD:
            self._validate_relationships(validated_values)
        
        return results
    
    def _validate_relationships(self, validated_params: Dict[str, mx.array]):
        """Validate relationships between parameters."""
        for param_names, constraint in self._relationship_constraints:
            if all(name in validated_params for name in param_names):
                values = [validated_params[name] for name in param_names]
                if not constraint.check(*values):
                    param_desc = ", ".join(param_names)
                    raise DistributionValidationError(
                        f"Relationship constraint violated for parameters ({param_desc}): {constraint.description}",
                        suggestions=constraint.suggest_fix("relationship", values)
                    )
    
    @staticmethod
    def _ensure_array(value: Any, dtype: Optional[mx.Dtype] = None) -> mx.array:
        """Enhanced array conversion with better error handling."""
        if dtype is None:
            dtype = mx.float32
        
        try:
            if isinstance(value, mx.array):
                return mx.astype(value, dtype) if value.dtype != dtype else value
            else:
                return mx.array(value, dtype=dtype)
        except Exception as e:
            raise DistributionValidationError(
                f"Cannot convert value to mx.array: {e}",
                suggestions=["Check that the value is numeric and finite"]
            )
    
    def compile_validator(self, param_name: str) -> Callable:
        """Compile validator for better performance."""
        if param_name in self._compiled_validators:
            return self._compiled_validators[param_name]
        
        # This is a placeholder for actual compilation
        # In practice, you'd use mx.compile or similar
        def compiled_validator(value):
            return self.validate_param(param_name, value)
        
        self._compiled_validators[param_name] = compiled_validator
        return compiled_validator
    
    def to_pytorch_constraints(self) -> Dict[str, Any]:
        """Export constraints to PyTorch format."""
        # Placeholder for cross-framework compatibility
        pytorch_constraints = {}
        for param_name, constraint in self._param_constraints.items():
            if isinstance(constraint, PositiveConstraint):
                pytorch_constraints[param_name] = "torch.distributions.constraints.positive"
            # Add more mappings as needed
        return pytorch_constraints
    
    def generate_documentation(self) -> str:
        """Auto-generate parameter documentation."""
        doc_lines = ["Parameters:"]
        for param_name, constraint in self._param_constraints.items():
            doc_lines.append(f"    {param_name}: {constraint}")
        
        if self._relationship_constraints:
            doc_lines.append("\nConstraints:")
            for param_names, constraint in self._relationship_constraints:
                param_list = ", ".join(param_names)
                doc_lines.append(f"    {param_list}: {constraint.description}")
        
        return "\n".join(doc_lines)


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
