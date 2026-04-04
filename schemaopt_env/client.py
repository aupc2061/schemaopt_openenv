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
            catalog_summary=obs_data.get("catalog_summary", {}),
            workload_summary=obs_data.get("workload_summary", {}),
            retrieval_context=obs_data.get("retrieval_context", {}),
            benchmark_context=obs_data.get("benchmark_context", {}),
            router_summary=obs_data.get("router_summary", {}),
            action_feedback=obs_data.get("action_feedback", {}),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(observation=observation, reward=observation.reward, done=observation.done)

    def _parse_state(self, payload: Dict) -> SchemaOptState:
        return SchemaOptState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            done=payload.get("done", False),
            task_id=payload.get("task_id", "spider::unknown"),
            difficulty=payload.get("difficulty", "spider"),
            derived_object_count=payload.get("derived_object_count", 0),
            checkpoint_count=payload.get("checkpoint_count", 0),
            retrieval_count=payload.get("retrieval_count", 0),
            benchmark_runs=payload.get("benchmark_runs", 0),
            storage_used_multiplier=payload.get("storage_used_multiplier", 0.0),
            final_score=payload.get("final_score"),
            last_error=payload.get("last_error"),
        )
