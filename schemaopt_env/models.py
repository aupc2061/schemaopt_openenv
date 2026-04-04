"""Models for the schema optimization environment.

This module prefers the real OpenEnv/Pydantic types when available, but falls back to
lightweight stdlib-backed models so the environment can still be smoke-tested locally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

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
        task_id: str = "spider::unknown"
        difficulty: str = "spider"
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
        task_id: str = "spider::unknown"
        difficulty: str = "spider"
        derived_object_count: int = 0
        checkpoint_count: int = 0
        retrieval_count: int = 0
        benchmark_runs: int = 0
        storage_used_multiplier: float = 0.0
        final_score: Optional[float] = None
        last_error: Optional[str] = None


@dataclass(frozen=True)
class TableSpec:
    """Static table metadata exposed to the environment."""

    name: str
    columns: tuple[tuple[str, str], ...]
    row_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": [{"name": name, "type": dtype} for name, dtype in self.columns],
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class QuerySpec:
    """A single workload query and its precomputed metadata."""

    query_id: str
    sql: str
    normalized_sql: str
    cluster_id: str
    business_tag: str
    frequency_weight: float
    priority_weight: float
    tables: tuple[str, ...]
    canonical_tables: tuple[str, ...]
    columns: tuple[str, ...]
    group_by: tuple[str, ...]
    filter_tokens: tuple[str, ...]
    filter_predicates: tuple[str, ...]
    canonical_filter_predicates: tuple[str, ...]
    measure_columns: tuple[str, ...]
    aggregate_functions: tuple[str, ...]
    plan_features: tuple[str, ...]
    description: str

    @property
    def weighted_cost(self) -> float:
        return round(self.frequency_weight * self.priority_weight, 6)

    @property
    def rewrite_template_hint(self) -> Dict[str, Any]:
        return {
            "canonical_source_tables": list(self.canonical_tables),
            "canonical_predicates": list(self.canonical_filter_predicates),
            "required_dimensions": list(self.group_by),
            "required_measures": list(self.measure_columns),
            "aggregate_functions": list(self.aggregate_functions),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "cluster_id": self.cluster_id,
            "business_tag": self.business_tag,
            "frequency_weight": self.frequency_weight,
            "priority_weight": self.priority_weight,
            "weighted_cost": self.weighted_cost,
            "tables": list(self.tables),
            "canonical_tables": list(self.canonical_tables),
            "columns": list(self.columns),
            "group_by": list(self.group_by),
            "filter_tokens": list(self.filter_tokens),
            "measure_columns": list(self.measure_columns),
            "aggregate_functions": list(self.aggregate_functions),
            "plan_features": list(self.plan_features),
            "description": self.description,
            "rewrite_template_hint": self.rewrite_template_hint,
        }

    def context(self, similar_ids: Sequence[str]) -> Dict[str, Any]:
        payload = self.summary()
        payload.update(
            {
                "sql": self.sql,
                "filter_predicates": list(self.filter_predicates),
                "canonical_filter_predicates": list(self.canonical_filter_predicates),
                "similar_query_ids": list(similar_ids),
                "suggested_exact_derived_shape": {
                    "object_kind": "agg_matview",
                    "source_objects": list(self.tables),
                    "group_by": list(self.group_by),
                    "canonical_predicates": list(self.canonical_filter_predicates),
                    "measure_columns": list(self.measure_columns),
                    "aggregate_functions": list(self.aggregate_functions),
                },
            }
        )
        return payload


@dataclass(frozen=True)
class ClusterSpec:
    """A workload cluster used for reset summaries and retrieval."""

    cluster_id: str
    label: str
    business_label: str
    query_ids: tuple[str, ...]
    query_count: int
    total_frequency_weight: float
    total_weighted_baseline_cost: float
    top_tables: tuple[str, ...]
    common_operator_patterns: tuple[str, ...]
    representative_dimensions: tuple[str, ...]
    representative_measures: tuple[str, ...]
    hotspot_rank: int
    preferred_object_kind: str
    representative_query_id: Optional[str] = None
    cluster_grain_emphasis: tuple[str, ...] = ()
    suggested_exact_derived_shape: Dict[str, Any] | None = None
    reference_rewrite_feasible: bool = True

    def to_summary(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "business_label": self.business_label,
            "query_count": self.query_count,
            "total_frequency_weight": self.total_frequency_weight,
            "total_weighted_baseline_cost": self.total_weighted_baseline_cost,
            "top_tables": list(self.top_tables),
            "common_operator_patterns": list(self.common_operator_patterns),
            "representative_dimensions": list(self.representative_dimensions),
            "representative_measures": list(self.representative_measures),
            "hotspot_rank": self.hotspot_rank,
            "preferred_object_kind": self.preferred_object_kind,
            "representative_query_id": self.representative_query_id,
            "cluster_grain_emphasis": list(self.cluster_grain_emphasis),
            "suggested_exact_derived_shape": dict(self.suggested_exact_derived_shape or {}),
            "reference_rewrite_feasible": self.reference_rewrite_feasible,
        }


@dataclass(frozen=True)
class TaskSpec:
    """A complete schema optimization episode definition."""

    task_id: str
    difficulty: str
    domain: str
    objective: str
    seed_source: str
    dataset_dir: str
    database_path: str
    tables: tuple[TableSpec, ...]
    visible_queries: tuple[QuerySpec, ...]
    holdout_queries: tuple[QuerySpec, ...]
    clusters: tuple[ClusterSpec, ...]
    budgets: Dict[str, Any]
    allowed_object_kinds: tuple[str, ...]
    engine_capabilities: Dict[str, Any]

    @property
    def total_visible_weighted_cost(self) -> float:
        return round(sum(query.weighted_cost for query in self.visible_queries), 6)

    def task_summary(self) -> Dict[str, Any]:
        return {
            "id": self.task_id,
            "difficulty": self.difficulty,
            "domain": self.domain,
            "objective": self.objective,
            "seed_source": self.seed_source,
            "dataset_dir": self.dataset_dir,
            "database_path": self.database_path,
            "visible_query_count": len(self.visible_queries),
            "holdout_query_count": len(self.holdout_queries),
            "cluster_count": len(self.clusters),
            "budgets": self.budgets,
            "allowed_object_kinds": list(self.allowed_object_kinds),
            "engine_capabilities": dict(self.engine_capabilities),
        }

    def reset_payload(self) -> Dict[str, Any]:
        return {
            "task": {
                "id": self.task_id,
                "objective": self.objective,
                "difficulty": self.difficulty,
                "domain": self.domain,
                "budgets": self.budgets,
                "allowed_object_kinds": list(self.allowed_object_kinds),
                "submission_rules": {
                    "query_sql_visible_at_reset": False,
                    "query_rewrites_allowed": False,
                    "holdout_workload_used_only_on_submit": True,
                },
                "seed_source": self.seed_source,
                "engine_capabilities": dict(self.engine_capabilities),
            },
            "catalog_summary": {
                "schemas": ["raw", "derived"],
                "tables": [table.to_dict() for table in self.tables],
                "lineage_edges": [],
                "derived_objects": [],
                "storage_usage_estimate": 0.0,
                "refresh_cost_estimate": 0.0,
            },
            "workload_summary": {
                "visible_query_count": len(self.visible_queries),
                "holdout_query_count": len(self.holdout_queries),
                "total_weighted_baseline_cost": self.total_visible_weighted_cost,
                "top_hotspot_clusters": [cluster.to_summary() for cluster in self.clusters[: min(3, len(self.clusters))]],
                "all_clusters": [cluster.to_summary() for cluster in self.clusters],
            },
        }


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
