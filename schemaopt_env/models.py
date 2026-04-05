"""Models for the schema optimization environment.

This module prefers the real OpenEnv/Pydantic types when available, but falls back to
lightweight stdlib-backed models so the environment can still be smoke-tested locally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

SchemaOptOperation = Literal[
    "inspect_catalog",
    "inspect_table_stats",
    "inspect_cluster",
    "inspect_query",
    "inspect_query_plan",
    "inspect_router_status",
    "retrieve_queries",
    "get_query_context",
    "create_derived_object",
    "modify_derived_object",
    "drop_derived_object",
    "list_derived_objects",
    "checkpoint",
    "revert_checkpoint",
    "benchmark_subset",
    "benchmark_cluster",
    "submit",
]
DerivedObjectKind = Literal["join_matview", "agg_matview", "filtered_projection", "denorm_table"]
ObservationStatus = Literal["ok", "error", "completed"]

try:
    from pydantic import Field, model_validator

    try:
        from openenv.core.env_server.types import Action, Observation, State
    except ImportError:
        from openenv_core.env_server.types import Action, Observation, State

    class SchemaOptAction(Action):
        """Action payload for workload-adaptive schema optimization."""

        operation: SchemaOptOperation
        target_id: Optional[str] = None
        query_ids: List[str] = Field(default_factory=list)
        pattern: Optional[str] = None
        cluster_id: Optional[str] = None
        tables: List[str] = Field(default_factory=list)
        columns: List[str] = Field(default_factory=list)
        plan_features: List[str] = Field(default_factory=list)
        top_k: Optional[int] = None
        object_kind: Optional[DerivedObjectKind] = None
        name: Optional[str] = None
        sql_definition: Optional[str] = None
        source_objects: List[str] = Field(default_factory=list)
        grain_hint: Optional[str] = None
        intended_clusters: List[str] = Field(default_factory=list)
        routing_tags: List[str] = Field(default_factory=list)

        @model_validator(mode="after")
        def validate_payload(self):
            _validate_action_payload(self)
            return self

    class SchemaOptObservation(Observation):
        """Observation returned from each schema optimization step."""

        status: ObservationStatus = "ok"
        message: str = ""
        catalog_summary: Dict[str, Any] = Field(default_factory=dict)
        workload_summary: Dict[str, Any] = Field(default_factory=dict)
        retrieval_context: Dict[str, Any] = Field(default_factory=dict)
        benchmark_context: Dict[str, Any] = Field(default_factory=dict)
        router_summary: Dict[str, Any] = Field(default_factory=dict)
        action_feedback: Dict[str, Any] = Field(default_factory=dict)

    class SchemaOptState(State):
        """State tracked across a single schema optimization episode."""

        step_count: int = 0
        done: bool = False
        task_id: str = "schemaopt_easy_hiring_pipeline"
        difficulty: str = "easy"
        derived_object_count: int = 0
        checkpoint_count: int = 0
        retrieval_count: int = 0
        benchmark_runs: int = 0
        storage_used_multiplier: float = 0.0
        final_score: Optional[float] = None
        last_error: Optional[str] = None

except ImportError:
    @dataclass
    class _BaseModel:
        def model_dump(self, exclude_none: bool = False) -> Dict[str, Any]:
            payload = asdict(self)
            if exclude_none:
                return {key: value for key, value in payload.items() if value is not None}
            return payload

        @classmethod
        def model_json_schema(cls) -> Dict[str, Any]:
            return {"title": cls.__name__}

    @dataclass
    class Action(_BaseModel):
        episode_id: Optional[str] = None

    @dataclass
    class Observation(_BaseModel):
        reward: float = 0.0
        done: bool = False
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class State(_BaseModel):
        episode_id: Optional[str] = None

    @dataclass
    class SchemaOptAction(Action):
        operation: SchemaOptOperation = "inspect_catalog"
        target_id: Optional[str] = None
        query_ids: List[str] = field(default_factory=list)
        pattern: Optional[str] = None
        cluster_id: Optional[str] = None
        tables: List[str] = field(default_factory=list)
        columns: List[str] = field(default_factory=list)
        plan_features: List[str] = field(default_factory=list)
        top_k: Optional[int] = None
        object_kind: Optional[DerivedObjectKind] = None
        name: Optional[str] = None
        sql_definition: Optional[str] = None
        source_objects: List[str] = field(default_factory=list)
        grain_hint: Optional[str] = None
        intended_clusters: List[str] = field(default_factory=list)
        routing_tags: List[str] = field(default_factory=list)

        def __post_init__(self):
            _validate_action_payload(self)

    @dataclass
    class SchemaOptObservation(Observation):
        status: ObservationStatus = "ok"
        message: str = ""
        catalog_summary: Dict[str, Any] = field(default_factory=dict)
        workload_summary: Dict[str, Any] = field(default_factory=dict)
        retrieval_context: Dict[str, Any] = field(default_factory=dict)
        benchmark_context: Dict[str, Any] = field(default_factory=dict)
        router_summary: Dict[str, Any] = field(default_factory=dict)
        action_feedback: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class SchemaOptState(State):
        step_count: int = 0
        done: bool = False
        task_id: str = "schemaopt_easy_hiring_pipeline"
        difficulty: str = "easy"
        derived_object_count: int = 0
        checkpoint_count: int = 0
        retrieval_count: int = 0
        benchmark_runs: int = 0
        storage_used_multiplier: float = 0.0
        final_score: Optional[float] = None
        last_error: Optional[str] = None


def _validate_action_payload(action: Any) -> None:
    op = action.operation
    if op in {"inspect_table_stats", "inspect_cluster", "inspect_query", "inspect_query_plan"} and not action.target_id:
        raise ValueError("target_id is required for the selected inspection action")
    if op == "retrieve_queries" and not any([
        action.pattern,
        action.cluster_id,
        action.tables,
        action.columns,
        action.plan_features,
        action.top_k,
    ]):
        raise ValueError("retrieve_queries requires at least one filter argument")
    if op == "get_query_context" and not action.query_ids:
        raise ValueError("query_ids is required for get_query_context")
    if op in {"create_derived_object", "modify_derived_object"}:
        if not action.object_kind:
            raise ValueError("object_kind is required for create/modify")
        if not action.name:
            raise ValueError("name is required for create/modify")
        if not action.sql_definition:
            raise ValueError("sql_definition is required for create/modify")
        if not action.source_objects:
            raise ValueError("source_objects is required for create/modify")
    if op == "drop_derived_object" and not action.target_id:
        raise ValueError("target_id is required for drop_derived_object")
    if op == "benchmark_subset" and not action.query_ids:
        raise ValueError("query_ids is required for benchmark_subset")
    if op == "benchmark_cluster" and not action.cluster_id:
        raise ValueError("cluster_id is required for benchmark_cluster")
