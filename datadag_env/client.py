"""Client implementation for DataDAG environment."""

from typing import Dict

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import DatadagAction, DatadagObservation, DatadagState
except ImportError:
    from openenv_core.client_types import StepResult
    from openenv_core.env_client import EnvClient

    from models import DatadagAction, DatadagObservation, DatadagState


class DatadagEnv(EnvClient[DatadagAction, DatadagObservation, DatadagState]):
    """Typed HTTP/WebSocket client for DataDAG."""

    def _step_payload(self, action: DatadagAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[DatadagObservation]:
        obs_data = payload.get("observation", {})

        observation = DatadagObservation(
            dag_integrity_status=obs_data.get("dag_integrity_status", "valid"),
            execution_trace=obs_data.get("execution_trace", ""),
            data_sample=obs_data.get("data_sample"),
            cascading_failure_nodes=obs_data.get("cascading_failure_nodes", []),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> DatadagState:
        return DatadagState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            done=payload.get("done", False),
            task_id=payload.get("task_id", "task1_easy"),
            dag_node_count=payload.get("dag_node_count", 0),
            last_error=payload.get("last_error"),
            final_score=payload.get("final_score"),
        )
