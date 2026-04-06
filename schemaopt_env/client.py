"""Client implementation for the schema optimization environment."""

from typing import Dict

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import SchemaOptAction, SchemaOptObservation, SchemaOptState
except ImportError:
    from openenv_core.client_types import StepResult
    from openenv_core.env_client import EnvClient

    from models import SchemaOptAction, SchemaOptObservation, SchemaOptState


class SchemaOptEnv(EnvClient[SchemaOptAction, SchemaOptObservation, SchemaOptState]):
    """Typed HTTP/WebSocket client for schema optimization episodes."""

    def _step_payload(self, action: SchemaOptAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[SchemaOptObservation]:
        obs_data = payload.get("observation", {})
        observation = SchemaOptObservation(
            status=obs_data.get("status", "ok"),
            message=obs_data.get("message", ""),
            decision_state=obs_data.get("decision_state", {}),
            catalog_summary=obs_data.get("catalog_summary", {}),
            workload_summary=obs_data.get("workload_summary", {}),
            retrieval_context=obs_data.get("retrieval_context", {}),
            benchmark_context=obs_data.get("benchmark_context", {}),
            router_summary=obs_data.get("router_summary", {}),
            action_feedback=obs_data.get("action_feedback", {}),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            done=payload.get("done", obs_data.get("done", False)),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(observation=observation, reward=observation.reward, done=observation.done)

    def _parse_state(self, payload: Dict) -> SchemaOptState:
        return SchemaOptState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            done=payload.get("done", False),
            task_id=payload.get("task_id", "schemaopt_easy_hiring_pipeline"),
            difficulty=payload.get("difficulty", "easy"),
            derived_object_count=payload.get("derived_object_count", 0),
            checkpoint_count=payload.get("checkpoint_count", 0),
            retrieval_count=payload.get("retrieval_count", 0),
            benchmark_runs=payload.get("benchmark_runs", 0),
            storage_used_multiplier=payload.get("storage_used_multiplier", 0.0),
            final_score=payload.get("final_score"),
            last_error=payload.get("last_error"),
            current_focus_cluster_id=payload.get("current_focus_cluster_id"),
            last_action_operation=payload.get("last_action_operation"),
            last_action_status=payload.get("last_action_status"),
            last_scope_key=payload.get("last_scope_key"),
            last_scope_benchmark_score=payload.get("last_scope_benchmark_score"),
            cluster_status_by_id=payload.get("cluster_status_by_id", {}),
            cluster_attempt_counts=payload.get("cluster_attempt_counts", {}),
            cluster_best_gated_improvement=payload.get("cluster_best_gated_improvement", {}),
            cluster_last_routed_query_count=payload.get("cluster_last_routed_query_count", {}),
            cluster_last_incorrect_query_count=payload.get("cluster_last_incorrect_query_count", {}),
            cluster_last_benchmark_score=payload.get("cluster_last_benchmark_score", {}),
            cluster_dominant_rejection_reason=payload.get("cluster_dominant_rejection_reason", {}),
            derived_object_names=payload.get("derived_object_names", []),
            useful_derived_object_names=payload.get("useful_derived_object_names", []),
            unused_derived_object_names=payload.get("unused_derived_object_names", []),
            remaining_steps=payload.get("remaining_steps", 0),
            remaining_object_budget=payload.get("remaining_object_budget", 0),
            remaining_storage_bytes=payload.get("remaining_storage_bytes", 0),
            remaining_refresh_runtime_ms=payload.get("remaining_refresh_runtime_ms", 0.0),
            resource_pressure=payload.get("resource_pressure", 0.0),
        )
