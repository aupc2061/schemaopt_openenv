"""Pydantic models for DataDAG environment."""

from typing import List, Literal, Optional
from pydantic import Field, model_validator
from openenv.core.env_server.types import Action, Observation, State


PipelineCommand = Literal[
    "create_node", "update_node", "execute_dag", "view_lineage"
]
DagIntegrityStatus = Literal["valid", "compilation_error", "circular_dependency"]


class DatadagAction(Action):
    """Action payload for pipeline operations."""

    pipeline_command: PipelineCommand
    model_identifier: Optional[str] = None
    sql_syntax: Optional[str] = None
    upstream_dependencies: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_command_payload(self):
        command = self.pipeline_command

        if command in ("create_node", "update_node"):
            if not self.model_identifier:
                raise ValueError("model_identifier is required for create/update")
            if not self.sql_syntax:
                raise ValueError("sql_syntax is required for create/update")

        if command in ("execute_dag", "view_lineage"):
            if self.sql_syntax is not None:
                raise ValueError("sql_syntax is not allowed for execute/view")
            if self.model_identifier is not None and command == "execute_dag":
                raise ValueError("model_identifier is not used for execute_dag")

        return self


class DatadagObservation(Observation):
    """Observation returned from each environment step."""

    dag_integrity_status: DagIntegrityStatus = "valid"
    execution_trace: str = ""
    data_sample: Optional[List[List]] = None
    cascading_failure_nodes: List[str] = Field(default_factory=list)


class DatadagState(State):
    """State tracked across a single episode."""

    step_count: int = 0
    done: bool = False
    task_id: str = "task1_easy"
    dag_node_count: int = 0
    last_error: Optional[str] = None
    final_score: Optional[float] = None
