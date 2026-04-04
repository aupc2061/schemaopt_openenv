"""Spider workload helpers and compatibility shims for schema optimization episodes."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from .spider_registry import (
        SpiderDatabaseSpec,
        SpiderQuerySpec,
        get_spider_database_spec,
        get_spider_database_summary,
        list_spider_database_summaries,
        spider_queries_for_db,
        sample_spider_queries_for_db,
    )
    from .models import ClusterSpec, QuerySpec, TableSpec, TaskSpec
except ImportError:
    from spider_registry import (  # type: ignore
        SpiderDatabaseSpec,
        SpiderQuerySpec,
        get_spider_database_spec,
        get_spider_database_summary,
        list_spider_database_summaries,
        spider_queries_for_db,
        sample_spider_queries_for_db,
    )
    from models import ClusterSpec, QuerySpec, TableSpec, TaskSpec  # type: ignore

_ALIAS_TOKEN_RE = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.((?:"[^"]+")|(?:[a-zA-Z_][a-zA-Z0-9_]*))')
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().lower().split())


def _canonicalize_table_name(table_name: str) -> str:
    return table_name.replace('"', '').strip().lower()


def _canonicalize_predicate(predicate: str) -> str:
    normalized = " ".join(predicate.strip().lower().split())
    normalized = _ALIAS_TOKEN_RE.sub(lambda match: match.group(1), normalized)
    normalized = normalized.replace('"', '')
    normalized = normalized.replace('( ', '(').replace(' )', ')')
    return normalized


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
    tables = tuple(payload.get("tables", []))
    filter_predicates = tuple(payload.get("filter_predicates", []))
    canonical_filter_predicates = tuple(_canonicalize_predicate(item) for item in filter_predicates)
    return QuerySpec(
        query_id=payload["query_id"],
        sql=sql,
        normalized_sql=payload.get("normalized_sql") or _normalize_sql(sql),
        cluster_id=payload["cluster_id"],
        business_tag=payload.get("business_tag", payload["cluster_id"]),
        frequency_weight=float(payload.get("frequency_weight", 1.0)),
        priority_weight=float(payload.get("priority_weight", 1.0)),
        tables=tables,
        canonical_tables=tuple(payload.get("canonical_tables") or [_canonicalize_table_name(item) for item in tables]),
        columns=tuple(column.lower() for column in payload.get("columns", [])),
        group_by=tuple(column.lower() for column in payload.get("group_by", [])),
        filter_tokens=tuple(token.lower() for token in payload.get("filter_tokens", [])),
        filter_predicates=filter_predicates,
        canonical_filter_predicates=canonical_filter_predicates,
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
        representative_query_id=payload.get("representative_query_id"),
        cluster_grain_emphasis=tuple(column.lower() for column in payload.get("cluster_grain_emphasis", [])),
        suggested_exact_derived_shape=dict(payload.get("suggested_exact_derived_shape", {})) if payload.get("suggested_exact_derived_shape") else None,
        reference_rewrite_feasible=bool(payload.get("reference_rewrite_feasible", True)),
    )


def _enrich_clusters(clusters: Sequence[ClusterSpec], visible_queries: Sequence[QuerySpec]) -> tuple[ClusterSpec, ...]:
    query_by_id = {query.query_id: query for query in visible_queries}
    enriched: List[ClusterSpec] = []
    for cluster in clusters:
        representative_query = query_by_id.get(cluster.representative_query_id or "")
        if representative_query is None:
            representative_query = next((query for query in visible_queries if query.cluster_id == cluster.cluster_id), None)
        suggested_shape = cluster.suggested_exact_derived_shape or (representative_query.context([]).get("suggested_exact_derived_shape") if representative_query else {})
        enriched.append(
            ClusterSpec(
                cluster_id=cluster.cluster_id,
                label=cluster.label,
                business_label=cluster.business_label,
                query_ids=cluster.query_ids,
                query_count=cluster.query_count,
                total_frequency_weight=cluster.total_frequency_weight,
                total_weighted_baseline_cost=cluster.total_weighted_baseline_cost,
                top_tables=cluster.top_tables,
                common_operator_patterns=cluster.common_operator_patterns,
                representative_dimensions=cluster.representative_dimensions,
                representative_measures=cluster.representative_measures,
                hotspot_rank=cluster.hotspot_rank,
                preferred_object_kind=cluster.preferred_object_kind,
                representative_query_id=representative_query.query_id if representative_query else cluster.representative_query_id,
                cluster_grain_emphasis=cluster.cluster_grain_emphasis or (representative_query.group_by if representative_query else ()),
                suggested_exact_derived_shape=suggested_shape,
                reference_rewrite_feasible=cluster.reference_rewrite_feasible,
            )
        )
    return tuple(enriched)


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


def list_spider_databases() -> List[Dict[str, Any]]:
    """Return Spider database summaries for migration-phase usage."""

    return list_spider_database_summaries()


def spider_database_summary(db_id: str) -> Dict[str, Any]:
    """Return a single Spider database summary."""

    return get_spider_database_summary(db_id)


def sample_spider_workload(
    db_id: str,
    sample_size: int,
    *,
    split: Optional[str] = None,
    seed: Optional[int] = None,
    min_complexity: float = 0.0,
) -> List[SpiderQuerySpec]:
    """Weighted sampling over Spider text-to-SQL examples for one database."""

    return sample_spider_queries_for_db(
        db_id=db_id,
        sample_size=sample_size,
        split=split,
        seed=seed,
        min_complexity=min_complexity,
    )


def build_spider_episode_task(
    db_id: str,
    *,
    split: str = "dev",
    sample_size: int = 24,
    holdout_size: int = 8,
    seed: Optional[int] = None,
    min_complexity: float = 0.0,
) -> TaskSpec:
    """Build a TaskSpec-compatible Spider episode for incremental migration.

    This keeps environment APIs stable while shifting workload selection to
    db_id-scoped Spider text-to-SQL samples.
    """

    db_spec = get_spider_database_spec(db_id)
    visible_spider = sample_spider_queries_for_db(
        db_id=db_id,
        sample_size=sample_size,
        split=split,
        seed=seed,
        min_complexity=min_complexity,
    )
    if not visible_spider:
        visible_spider = sample_spider_queries_for_db(
            db_id=db_id,
            sample_size=sample_size,
            split=None,
            seed=seed,
            min_complexity=min_complexity,
        )

    visible_ids = {query.query_id for query in visible_spider}
    fallback_pool = [query for query in spider_queries_for_db(db_id, split=split) if query.query_id not in visible_ids]
    if len(fallback_pool) < holdout_size:
        fallback_pool = [query for query in spider_queries_for_db(db_id) if query.query_id not in visible_ids]
    holdout_spider = fallback_pool[: max(0, holdout_size)]

    def _to_query_spec(query: SpiderQuerySpec) -> QuerySpec:
        cluster_id = f"spider_{query.db_id}_all"
        return QuerySpec(
            query_id=query.query_id,
            sql=query.sql,
            normalized_sql=query.normalized_sql,
            cluster_id=cluster_id,
            business_tag=query.db_id,
            frequency_weight=max(0.01, query.sampling_weight),
            priority_weight=1.0,
            tables=query.tables,
            canonical_tables=tuple(table.lower() for table in query.tables),
            columns=(),
            group_by=(),
            filter_tokens=tuple(query.operation_patterns),
            filter_predicates=(),
            canonical_filter_predicates=(),
            measure_columns=(),
            aggregate_functions=(),
            plan_features=tuple(query.operation_patterns),
            description=query.question or query.query_id,
        )

    visible_queries = tuple(_to_query_spec(query) for query in visible_spider)
    holdout_queries = tuple(_to_query_spec(query) for query in holdout_spider)

    table_specs = tuple(TableSpec(name=table, columns=(), row_count=0) for table in db_spec.table_names)
    cluster_id = f"spider_{db_id}_all"
    cluster = ClusterSpec(
        cluster_id=cluster_id,
        label=f"Spider workload for {db_id}",
        business_label=db_id,
        query_ids=tuple(query.query_id for query in visible_queries),
        query_count=len(visible_queries),
        total_frequency_weight=round(sum(query.frequency_weight for query in visible_queries), 6),
        total_weighted_baseline_cost=round(sum(query.weighted_cost for query in visible_queries), 6),
        top_tables=tuple(table_specs[idx].name for idx in range(min(4, len(table_specs)))),
        common_operator_patterns=tuple(sorted({pattern for query in visible_spider for pattern in query.operation_patterns})),
        representative_dimensions=(),
        representative_measures=(),
        hotspot_rank=1,
        preferred_object_kind="agg_matview",
        representative_query_id=visible_queries[0].query_id if visible_queries else None,
        cluster_grain_emphasis=(),
        suggested_exact_derived_shape={
            "object_kind": "agg_matview",
            "source_objects": list(visible_queries[0].tables) if visible_queries else [],
            "group_by": [],
            "canonical_predicates": [],
            "measure_columns": [],
            "aggregate_functions": [],
        },
        reference_rewrite_feasible=True,
    )

    spider_root = str((Path(__file__).resolve().parent / "spider_data").resolve())
    return TaskSpec(
        task_id=f"spider::{split}::{db_id}",
        difficulty="spider",
        domain=db_id,
        objective="Optimize weighted historical SQL workload for Spider database",
        seed_source=spider_root,
        dataset_dir=spider_root,
        database_path=db_spec.sqlite_path,
        tables=table_specs,
        visible_queries=visible_queries,
        holdout_queries=holdout_queries,
        clusters=(cluster,),
        budgets={
            "max_new_derived_objects": 8,
            "max_storage_bytes": 4_000_000,
            "max_refresh_runtime_ms": 1_500,
        },
        allowed_object_kinds=("agg_matview", "join_matview", "denorm_table"),
        engine_capabilities={
            "dialect": "sqlite",
            "spider_mode": True,
            "db_id": db_id,
            "split": split,
            "weighted_sampling": True,
        },
    )


__all__ = [
    "TableSpec",
    "QuerySpec",
    "ClusterSpec",
    "TaskSpec",
    "SpiderDatabaseSpec",
    "SpiderQuerySpec",
    "query_lookup",
    "similar_query_ids",
    "cluster_lookup",
    "visible_queries_for_cluster",
    "match_queries",
    "load_catalog_from_duckdb",
    "list_spider_databases",
    "spider_database_summary",
    "sample_spider_workload",
    "build_spider_episode_task",
]
