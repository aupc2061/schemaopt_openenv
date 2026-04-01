"""Manifest-backed task catalog for the schema optimization benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASK_ASSET_ROOT = Path(__file__).resolve().parent / "task_assets"


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
    columns: tuple[str, ...]
    group_by: tuple[str, ...]
    filter_tokens: tuple[str, ...]
    filter_predicates: tuple[str, ...]
    measure_columns: tuple[str, ...]
    aggregate_functions: tuple[str, ...]
    plan_features: tuple[str, ...]
    description: str

    @property
    def weighted_cost(self) -> float:
        return round(self.frequency_weight * self.priority_weight, 6)

    def summary(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "cluster_id": self.cluster_id,
            "business_tag": self.business_tag,
            "frequency_weight": self.frequency_weight,
            "priority_weight": self.priority_weight,
            "weighted_cost": self.weighted_cost,
            "tables": list(self.tables),
            "columns": list(self.columns),
            "group_by": list(self.group_by),
            "filter_tokens": list(self.filter_tokens),
            "measure_columns": list(self.measure_columns),
            "aggregate_functions": list(self.aggregate_functions),
            "plan_features": list(self.plan_features),
            "description": self.description,
        }

    def context(self, similar_ids: Sequence[str]) -> Dict[str, Any]:
        payload = self.summary()
        payload.update(
            {
                "sql": self.sql,
                "filter_predicates": list(self.filter_predicates),
                "similar_query_ids": list(similar_ids),
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


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().lower().split())


def _resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def _load_table(payload: Dict[str, Any]) -> TableSpec:
    return TableSpec(
        name=payload["name"],
        columns=tuple((column["name"], column["type"]) for column in payload.get("columns", [])),
        row_count=int(payload.get("row_count") or 0),
    )


def _load_query(payload: Dict[str, Any]) -> QuerySpec:
    sql = payload["sql"]
    return QuerySpec(
        query_id=payload["query_id"],
        sql=sql,
        normalized_sql=payload.get("normalized_sql") or _normalize_sql(sql),
        cluster_id=payload["cluster_id"],
        business_tag=payload.get("business_tag", payload["cluster_id"]),
        frequency_weight=float(payload.get("frequency_weight", 1.0)),
        priority_weight=float(payload.get("priority_weight", 1.0)),
        tables=tuple(payload.get("tables", [])),
        columns=tuple(column.lower() for column in payload.get("columns", [])),
        group_by=tuple(column.lower() for column in payload.get("group_by", [])),
        filter_tokens=tuple(token.lower() for token in payload.get("filter_tokens", [])),
        filter_predicates=tuple(payload.get("filter_predicates", [])),
        measure_columns=tuple(column.lower() for column in payload.get("measure_columns", [])),
        aggregate_functions=tuple(function.lower() for function in payload.get("aggregate_functions", [])),
        plan_features=tuple(feature.lower() for feature in payload.get("plan_features", [])),
        description=payload.get("description", payload["query_id"]),
    )


def _load_cluster(payload: Dict[str, Any]) -> ClusterSpec:
    return ClusterSpec(
        cluster_id=payload["cluster_id"],
        label=payload.get("label", payload["cluster_id"]),
        business_label=payload.get("business_label", payload.get("label", payload["cluster_id"])),
        query_ids=tuple(payload.get("query_ids", [])),
        query_count=int(payload.get("query_count") or 0),
        total_frequency_weight=float(payload.get("total_frequency_weight", 0.0)),
        total_weighted_baseline_cost=float(payload.get("total_weighted_baseline_cost", 0.0)),
        top_tables=tuple(payload.get("top_tables", [])),
        common_operator_patterns=tuple(feature.lower() for feature in payload.get("common_operator_patterns", [])),
        representative_dimensions=tuple(column.lower() for column in payload.get("representative_dimensions", [])),
        representative_measures=tuple(column.lower() for column in payload.get("representative_measures", [])),
        hotspot_rank=int(payload.get("hotspot_rank") or 0),
        preferred_object_kind=payload.get("preferred_object_kind", "agg_matview"),
    )


def load_task_manifest(task_manifest_path: str | Path) -> TaskSpec:
    path = Path(task_manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset_dir = _resolve_repo_path(payload["dataset_dir"])
    database_path = _resolve_repo_path(payload["database_path"])
    tables = tuple(_load_table(item) for item in payload.get("tables", []))
    visible_queries = tuple(_load_query(item) for item in payload.get("visible_queries", []))
    holdout_queries = tuple(_load_query(item) for item in payload.get("holdout_queries", []))
    clusters = tuple(_load_cluster(item) for item in payload.get("clusters", []))
    return TaskSpec(
        task_id=payload["task_id"],
        difficulty=payload["difficulty"],
        domain=payload["domain"],
        objective=payload["objective"],
        seed_source=_resolve_repo_path(payload.get("seed_source", payload["dataset_dir"])),
        dataset_dir=dataset_dir,
        database_path=database_path,
        tables=tables,
        visible_queries=visible_queries,
        holdout_queries=holdout_queries,
        clusters=clusters,
        budgets=dict(payload.get("budgets", {})),
        allowed_object_kinds=tuple(payload.get("allowed_object_kinds", [])),
        engine_capabilities=dict(payload.get("engine_capabilities", {})),
    )


def discover_task_manifests(task_asset_root: str | Path = _TASK_ASSET_ROOT) -> Dict[str, TaskSpec]:
    root = Path(task_asset_root)
    if not root.exists():
        return {}

    tasks: Dict[str, TaskSpec] = {}
    for manifest_path in sorted(root.glob("*.json")):
        task = load_task_manifest(manifest_path)
        tasks[task.task_id] = task
    return tasks


TASK_CATALOG = discover_task_manifests()


def list_task_summaries() -> List[Dict[str, Any]]:
    return [task.task_summary() for task in TASK_CATALOG.values()]


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASK_CATALOG:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASK_CATALOG[task_id]


def query_lookup(task: TaskSpec, include_holdout: bool = True) -> Dict[str, QuerySpec]:
    lookup = {query.query_id: query for query in task.visible_queries}
    if include_holdout:
        lookup.update({query.query_id: query for query in task.holdout_queries})
    return lookup


def similar_query_ids(task: TaskSpec, query_id: str) -> List[str]:
    lookup = query_lookup(task)
    query = lookup[query_id]
    return [
        candidate.query_id
        for candidate in task.visible_queries
        if candidate.cluster_id == query.cluster_id and candidate.query_id != query_id
    ][:5]


def cluster_lookup(task: TaskSpec) -> Dict[str, ClusterSpec]:
    return {cluster.cluster_id: cluster for cluster in task.clusters}


def visible_queries_for_cluster(task: TaskSpec, cluster_id: str) -> List[QuerySpec]:
    return [query for query in task.visible_queries if query.cluster_id == cluster_id]


def _sorted_matches(matches: Iterable[QuerySpec], top_k: Optional[int]) -> List[QuerySpec]:
    ordered = sorted(matches, key=lambda query: query.query_id)
    if top_k is None:
        return ordered
    return ordered[:top_k]


def match_queries(
    task: TaskSpec,
    mode: str,
    pattern: str | None = None,
    cluster_id: str | None = None,
    tables: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    plan_features: Sequence[str] | None = None,
    top_k: int | None = None,
) -> List[QuerySpec]:
    """Deterministic retrieval over the visible workload."""

    queries = list(task.visible_queries)
    pattern_lc = (pattern or "").lower()
    tables_lc = {value.lower() for value in (tables or [])}
    columns_lc = {value.lower() for value in (columns or [])}
    plan_features_lc = {value.lower() for value in (plan_features or [])}

    if mode == "regex":
        if not pattern_lc:
            return []
        import re

        compiled = re.compile(pattern_lc)
        matches = [query for query in queries if compiled.search(query.normalized_sql)]
        return _sorted_matches(matches, top_k)
    if mode == "substring":
        if not pattern_lc:
            return []
        matches = [query for query in queries if pattern_lc in query.normalized_sql]
        return _sorted_matches(matches, top_k)
    if mode == "cluster":
        matches = [query for query in queries if query.cluster_id == cluster_id]
        return _sorted_matches(matches, top_k)
    if mode == "table_filter":
        matches = [query for query in queries if tables_lc and tables_lc.issubset(set(table.lower() for table in query.tables))]
        return _sorted_matches(matches, top_k)
    if mode == "column_filter":
        matches = [query for query in queries if columns_lc and columns_lc.issubset(set(query.columns))]
        return _sorted_matches(matches, top_k)
    if mode == "plan_filter":
        matches = [query for query in queries if plan_features_lc and plan_features_lc.issubset(set(query.plan_features))]
        return _sorted_matches(matches, top_k)
    if mode == "hotspot_rank":
        ordered = sorted(queries, key=lambda query: query.frequency_weight * query.priority_weight, reverse=True)
        return ordered[: max(1, top_k or 5)]
    raise ValueError(f"Unsupported retrieval mode: {mode}")


def load_catalog_from_duckdb(db_path: str) -> List[TableSpec]:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency-bound path
        raise RuntimeError("duckdb is required to inspect a real task database") from exc

    con = duckdb.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
            """
        ).fetchall()
        catalog: List[TableSpec] = []
        for schema_name, table_name in rows:
            columns = con.execute(
                f"DESCRIBE {schema_name}.{table_name}"
            ).fetchall()
            row_count = int(con.execute(f"SELECT COUNT(*) FROM {schema_name}.{table_name}").fetchone()[0])
            catalog.append(
                TableSpec(
                    name=f"{schema_name}.{table_name}",
                    columns=tuple((str(row[0]), str(row[1])) for row in columns),
                    row_count=row_count,
                )
            )
        return catalog
    finally:
        con.close()
