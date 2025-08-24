import mlx.core as mx
from typing import Optional, Any, Callable, Dict, List
import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CheckResult: 
    ok: mx.array                   # shape () boolean, device-side
    violations: mx.array | None    # boolean mask (same shape as value) or None
    summary: Dict[str, mx.array]   # small scalars/arrays, device-side

class Constraint: 
    def check(self, value: mx.array) -> CheckResult:
        raise NotImplementedError
    
    def error_message(self, param_name: str, actual_summary: Dict[str, any] | None = None) -> str: 
        raise NotImplementedError
    
    def describe(self) -> str:
        raise NotImplementedError
    
    def __call__(self, value: mx.array) -> CheckResult: 
        return self.check(value)
    
    def __str__(self):
        return self.describe()
    

@dataclass(frozen=True)
class Real(Constraint):
    def check(self, value: mx.array) -> CheckResult:
        finite = mx.isfinite(value)
        ok = mx.all(finite)
        num_bad = mx.size(value) - mx.sum(finite)
        return CheckResult(ok=ok, violations=~finite, summary={"num_bad": num_bad})
    
    def describe(self) -> str: 
        return "real (finite)"
    
    def error_message(self, param_name: str) -> str:
        return f"Parameter '{param_name}' must be finite (non NaN/Inf)"
    
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
    min_value: float = 1e-6
    def check(self, value: mx.array) -> CheckResult:
        mask = value > self.min_value
        ok = mx.all(mask)
        summary = {"min": mx.min(value), "num_bad": mx.size(value) - mx.sum(mask)}
        return CheckResult(ok=ok, violations=~mask, summary=summary)
    
    def describe(self) -> str:
        return "positive" if self.min_value == 0 else f"positive(> {self.min_value})"
    
    def error_message(self, param_name: str) -> str:
        return f"Parameter '{param_name}' must be {self.describe()}"
    
    def suggest_fix(self, param_name: str) -> List[str]:
        return [f"Ensure {param_name} > {self.min_value}"]

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
        lo = (value >= self.lower) if self.lower_inclusive else (value > self.lower)
        hi = (value <= self.upper) if self.upper_inclusive else (value < self.upper)
        mask = lo & hi
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
    # Works for integer dtypes, or floats that are whole within atol
        is_int_dtype = value.dtype in (mx.int8, mx.int16, mx.int32, mx.int64, getattr(mx, 'uint32', mx.int32))
        if is_int_dtype:
            ok = mx.array(True)
            return CheckResult(ok=ok, violations=None, summary={})
    
        # float case: value == round(value)
        rounded = mx.round(value)
        mask = mx.abs(value - rounded) <= 0.0
        ok = mx.all(mask)
        return CheckResult(ok=ok, violations=~mask, summary={"num_bad": mx.size(value) - mx.sum(mask)})

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
                 constraint: 'Constraint' | None = None,
                 suggestions: List[str] | None = None):
        super().__init__(message)
        self.param_name = param_name
        self.constaint = constraint
        self.suggestions = suggestions or []
    
    def __str__(self) -> str: 
        base = super().__str__()
        if self.param_name:
            base = f"[{self.param_name}] {base}"
        return base


class DistributionValidator:
    def __init__(self):
        self.should_validate: bool = True
        self._param_constraints: Dict[str, Constraint] = {}

    def register_constraint(self, param_name: str, constraint: Constraint):
        self._param_constraints[param_name] = constraint

    def validate_params(self, **params) -> Dict[str, ValidationResult]:
        if not self.should_validate:
            return {k: ValidationResult(v, []) for k, v in params.items()}

        arrays = {k: self._ensure_array(v) for k, v in params.items()}
        results: Dict[str, ValidationResult] = {}

        # Per-parameter constraints
        for name, x in arrays.items():
            res_warnings: List[str] = []
            c = self._param_constraints.get(name)
            if not isinstance(x, mx.array):
                raise DistributionValidationError("Expected mx.array", param_name=name)
            
            cres = c.check(x)
            if not bool(mx.asarray(cres.ok).item()):
                raise DistributionValidationError(
                    c.error_message(name, self._summarize(x)),
                    param_name=name,
                    constraint=c,
                    suggestions=c.suggest_fix(name, self._summarize(x))
                )
            res_warnings.append(f"{name}: {c.describe()} ok")
            results[name] = ValidationResult(x, res_warnings)

        return results

    @staticmethod
    def _summarize(x: mx.array) -> Dict[str, Any]:
        return {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "min": float(mx.min(x).item()),
            "max": float(mx.max(x).item()),
        }

    @staticmethod
    def _summarize_py(x: Any) -> Dict[str, Any]:
        try:
            if isinstance(x, mx.array):
                return DistributionValidator._summarize(x)
            return {"type": type(x).__name__}
        except Exception:
            return {"type": type(x).__name__}

    @staticmethod
    def _ensure_array(value: Any, dtype: Optional[mx.Dtype] = None) -> mx.array:
        if isinstance(value, mx.array):
            return value if dtype is None else mx.astype(value, dtype)
        if isinstance(value, bool):
            return mx.array(value, dtype or mx.bool_)
        if isinstance(value, int):
            return mx.array(value, dtype or mx.int32)
        if isinstance(value, float):
            return mx.array(value, dtype or mx.float32)
        return mx.array(value, dtype=dtype)

