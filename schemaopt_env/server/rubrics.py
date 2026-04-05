"""Rubrics for schema optimization reward computation.

Implements an OpenEnv-compatible rubric tree for ``schemaopt_env`` while keeping
all database execution inside the environment. Rubrics consume the structured
``action_feedback`` payload attached to observations.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from openenv.core.rubrics.base import Rubric
except ImportError:
    try:
        from openenv_core.rubrics.base import Rubric
    except ImportError:
        class Rubric:
            """Compatibility fallback when OpenEnv rubrics are unavailable."""

            def __init__(self) -> None:
                self.last_score: float | None = None
                self.last_details: Dict[str, Any] = {}
                self._rubric_children: Dict[str, "Rubric"] = {}

            def __setattr__(self, name: str, value: Any) -> None:
                if isinstance(value, Rubric):
                    self._rubric_children[name] = value
                object.__setattr__(self, name, value)

            def __call__(self, action: Any, observation: Any) -> float:
                result = self.forward(action, observation)
                self.last_score = result
                return result

            def forward(self, action: Any, observation: Any) -> float:
                raise NotImplementedError

            def reset(self) -> None:
                pass

            def named_rubrics(self, prefix: str = ""):
                for name, child in self._rubric_children.items():
                    full_name = f"{prefix}.{name}" if prefix else name
                    yield full_name, child
                    yield from child.named_rubrics(full_name)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _round_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, float):
            result[key] = round(value, 6)
        else:
            result[key] = value
    return result


class NoOpRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        self.last_details = {"noop_reward": 0.0, "total_reward": 0.0}
        return 0.0


class ErrorRubric(Rubric):
    def __init__(self) -> None:
        super().__init__()
        self.penalties = {
            "validation_error": -0.02,
            "sql_runtime_error": -0.03,
            "internal_error": -0.05,
        }

    def forward(self, action: Any, observation: Any) -> float:
        feedback = getattr(observation, "action_feedback", {}) or {}
        error_type = feedback.get("error_type") or "internal_error"
        penalty = float(self.penalties.get(error_type, -0.05))
        self.last_details = {
            "error_penalty": round(penalty, 6),
            "error_type": error_type,
            "total_reward": round(penalty, 6),
        }
        return penalty


class CreateModifyRubric(Rubric):
    def _object_utility(self, inputs: Dict[str, Any], prefix: str = "") -> float:
        visible_query_count = max(1, int(inputs.get(f"{prefix}visible_query_count", 1)))
        visible_cluster_count = max(1, int(inputs.get(f"{prefix}visible_cluster_count", 1)))
        eligible_visible_queries = float(inputs.get(f"{prefix}eligible_visible_queries", 0))
        eligible_cluster_count = float(inputs.get(f"{prefix}eligible_visible_cluster_count", 0))
        eligible_query_ratio = eligible_visible_queries / visible_query_count
        eligible_cluster_ratio = eligible_cluster_count / visible_cluster_count
        return (0.60 * eligible_query_ratio) + (0.40 * eligible_cluster_ratio)

    def forward(self, action: Any, observation: Any) -> float:
        inputs = (getattr(observation, "action_feedback", {}) or {}).get("reward_inputs", {})
        if inputs.get("duplicate_signature"):
            score = -0.02
            self.last_details = {
                "utility_component": 0.0,
                "resource_component": 0.0,
                "duplicate_penalty": -0.02,
                "empty_object_penalty": 0.0,
                "total_reward": round(score, 6),
            }
            return score
        if inputs.get("is_empty_object"):
            score = -0.03
            self.last_details = {
                "utility_component": 0.0,
                "resource_component": 0.0,
                "duplicate_penalty": 0.0,
                "empty_object_penalty": -0.03,
                "total_reward": round(score, 6),
            }
            return score

        resource_delta = max(0.0, float(inputs.get("resource_pressure_delta", 0.0)))
        operation = getattr(action, "operation", "")
        if operation == "modify_derived_object":
            current_utility = self._object_utility(inputs)
            previous_utility = self._object_utility(inputs, prefix="previous_")
            utility_component = 0.04 * (current_utility - previous_utility)
            resource_component = -0.03 * resource_delta
            score = _clamp(utility_component + resource_component, -0.04, 0.04)
        else:
            current_utility = self._object_utility(inputs)
            utility_component = 0.03 * current_utility
            resource_component = -0.02 * resource_delta
            score = _clamp(utility_component + resource_component, -0.03, 0.03)

        self.last_details = {
            "utility_component": round(utility_component, 6),
            "resource_component": round(resource_component, 6),
            "duplicate_penalty": 0.0,
            "empty_object_penalty": 0.0,
            "total_reward": round(score, 6),
        }
        return score


class DropRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        inputs = (getattr(observation, "action_feedback", {}) or {}).get("reward_inputs", {})
        is_empty = bool(inputs.get("is_empty_object"))
        duplicate_like = bool(inputs.get("duplicate_like"))
        eligible_visible_queries = int(inputs.get("eligible_visible_queries", 0))
        used_by_visible_queries = int(inputs.get("used_by_visible_queries_count", 0))
        resource_delta = float(inputs.get("resource_pressure_delta", 0.0))

        if is_empty or duplicate_like or (eligible_visible_queries == 0 and used_by_visible_queries == 0):
            utility_component = 0.005
        elif used_by_visible_queries == 0:
            utility_component = 0.0
        else:
            utility_component = -0.01

        resource_component = 0.01 * max(0.0, -resource_delta)
        score = _clamp(utility_component + resource_component, -0.02, 0.02)
        self.last_details = {
            "utility_component": round(utility_component, 6),
            "resource_component": round(resource_component, 6),
            "duplicate_penalty": 0.0,
            "empty_object_penalty": 0.0,
            "total_reward": round(score, 6),
        }
        return score


class BenchmarkRubric(Rubric):
    def __init__(self) -> None:
        super().__init__()
        self._scope_memory: Dict[str, Dict[str, float | str]] = {}

    def _score(self, inputs: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
        gated_improvement_component = 0.75 * float(inputs.get("gated_improvement", 0.0))
        routing_component = 0.15 * float(inputs.get("routed_query_ratio", 0.0))
        correctness_component = 0.10 * float(inputs.get("correctness_coverage", 1.0))
        budget_component = -0.50 * float(inputs.get("budget_penalty", 0.0))
        resource_component = -0.10 * float(inputs.get("resource_pressure", 0.0))
        incorrect_component = -0.05 * float(inputs.get("incorrect_query_ratio", 0.0))
        components = {
            "gated_improvement_component": gated_improvement_component,
            "routing_component": routing_component,
            "correctness_component": correctness_component,
            "budget_component": budget_component,
            "resource_component": resource_component,
            "incorrect_component": incorrect_component,
        }
        return sum(components.values()), components

    def forward(self, action: Any, observation: Any) -> float:
        feedback = getattr(observation, "action_feedback", {}) or {}
        inputs = feedback.get("reward_inputs", {})
        scope_key = str(feedback.get("scope_key") or "unknown")
        derived_state_hash = str(inputs.get("derived_state_hash") or "")
        previous = self._scope_memory.get(scope_key)
        if previous and previous.get("derived_state_hash") == derived_state_hash:
            previous_score = float(previous.get("benchmark_score", 0.0))
            self.last_details = {
                "gated_improvement_component": 0.0,
                "routing_component": 0.0,
                "correctness_component": 0.0,
                "budget_component": 0.0,
                "resource_component": 0.0,
                "incorrect_component": 0.0,
                "scope_previous_score": round(previous_score, 6),
                "scope_current_score": round(previous_score, 6),
                "total_reward": 0.0,
            }
            return 0.0

        current_score, components = self._score(inputs)
        previous_score = float(previous.get("benchmark_score", 0.0)) if previous else 0.0
        reward_delta = current_score if previous is None else current_score - previous_score
        reward_delta = _clamp(reward_delta, -0.20, 0.20)
        self._scope_memory[scope_key] = {
            "derived_state_hash": derived_state_hash,
            "benchmark_score": current_score,
            "gated_improvement": float(inputs.get("gated_improvement", 0.0)),
            "correctness_coverage": float(inputs.get("correctness_coverage", 1.0)),
            "routed_query_ratio": float(inputs.get("routed_query_ratio", 0.0)),
            "resource_pressure": float(inputs.get("resource_pressure", 0.0)),
        }
        self.last_details = _round_dict({
            **components,
            "scope_previous_score": previous_score,
            "scope_current_score": current_score,
            "total_reward": reward_delta,
        })
        return reward_delta

    def reset(self) -> None:
        self._scope_memory = {}


class SubmitRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        feedback = getattr(observation, "action_feedback", {}) or {}
        inputs = feedback.get("reward_inputs", {})
        score = float(inputs.get("final_score", feedback.get("final_score", 0.0)))
        final_inputs = feedback.get("final_score_inputs", {}) or inputs
        self.last_details = _round_dict({
            "final_score": score,
            **{key: float(value) if isinstance(value, (int, float)) else value for key, value in final_inputs.items()},
            "total_reward": score,
        })
        return score


class ActionDispatchRubric(Rubric):
    def __init__(self) -> None:
        super().__init__()
        self.create_modify = CreateModifyRubric()
        self.drop = DropRubric()
        self.benchmark = BenchmarkRubric()
        self.submit = SubmitRubric()
        self.noop = NoOpRubric()

    def forward(self, action: Any, observation: Any) -> float:
        operation = getattr(action, "operation", "")
        if operation in {"create_derived_object", "modify_derived_object"}:
            return self.create_modify(action, observation)
        if operation == "drop_derived_object":
            return self.drop(action, observation)
        if operation in {"benchmark_cluster", "benchmark_subset"}:
            return self.benchmark(action, observation)
        if operation == "submit":
            return self.submit(action, observation)
        return self.noop(action, observation)

    def reset(self) -> None:
        self.benchmark.reset()


class SchemaOptRubric(Rubric):
    """Root rubric for schema optimization reward computation."""

    def __init__(self) -> None:
        super().__init__()
        self.action_dispatch = ActionDispatchRubric()
        self.error = ErrorRubric()

    def forward(self, action: Any, observation: Any) -> float:
        feedback = getattr(observation, "action_feedback", {}) or {}
        if getattr(observation, "status", "ok") == "error":
            score = self.error(action, observation)
            feedback["reward_breakdown"] = {
                "source": "error",
                "total_reward": round(score, 6),
                "components": dict(getattr(self.error, "last_details", {})),
            }
            return score

        score = self.action_dispatch(action, observation)
        operation = getattr(action, "operation", "")
        component_details = dict(getattr(getattr(self.action_dispatch, "noop"), "last_details", {}))
        if operation in {"create_derived_object", "modify_derived_object"}:
            component_details = dict(getattr(self.action_dispatch.create_modify, "last_details", {}))
        elif operation == "drop_derived_object":
            component_details = dict(getattr(self.action_dispatch.drop, "last_details", {}))
        elif operation in {"benchmark_cluster", "benchmark_subset"}:
            component_details = dict(getattr(self.action_dispatch.benchmark, "last_details", {}))
        elif operation == "submit":
            component_details = dict(getattr(self.action_dispatch.submit, "last_details", {}))
        feedback["reward_breakdown"] = {
            "source": feedback.get("event"),
            "total_reward": round(score, 6),
            "components": component_details,
        }
        return score

    def reset(self) -> None:
        self.action_dispatch.reset()
