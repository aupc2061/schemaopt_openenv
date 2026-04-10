"""Manifest-backed task catalog for the schema optimization benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent
_TASK_ASSET_ROOTS = (
    Path(__file__).resolve().parent / "task_assets",
)
_ALIAS_TOKEN_RE = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.((?:"[^"]+")|(?:[a-zA-Z_][a-zA-Z0-9_]*))')
_GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_COMMON_RUNTIME_ROOTS = (
    "/app",
    "/workspace",
    "/tmp/workspace",
)


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


def _canonicalize_measure_name(value: str) -> str:
    normalized = value.strip().lower().replace('"', '')
    normalized = re.sub(r'\s+', ' ', normalized)
    if normalized in {"count(*)", "count_star()", "count_star"}:
        return "count_star"
    match = re.fullmatch(r'(count|sum|avg|min|max)\((.+)\)', normalized)
    if match:
        func = match.group(1)
        target = match.group(2).strip()
        if target == "*":
            return "count_star"
        target = target.replace('.', '_')
        target = re.sub(r'[^a-z0-9_]+', '_', target).strip('_')
        return f"{func}_{target}" if target else func
    normalized = normalized.replace('.', '_')
    normalized = re.sub(r'[^a-z0-9_]+', '_', normalized).strip('_')
    return normalized


def _default_result_label(value: str) -> str:
    normalized = value.strip().lower().replace('"', '')
    normalized = re.sub(r'\s+', ' ', normalized)
    if normalized == 'count(*)':
        return 'count_star()'
    return normalized


def _split_sql_list(clause: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for char in clause:
        if char == '(':
            depth += 1
        elif char == ')':
            depth = max(0, depth - 1)
        if char == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append(''.join(current).strip())
    return parts


def _extract_alias(expression: str) -> str | None:
    match = re.search(r'\bas\s+(?:"([^"]+)"|([A-Za-z_][A-Za-z0-9_]*))\s*$', expression, re.IGNORECASE)
    if not match:
        return None
    return (match.group(1) or match.group(2) or '').lower()


def _result_columns_from_sql(sql: str) -> tuple[str, ...]:
    normalized = sql.strip().rstrip(';')
    lowered = normalized.lower()
    select_start = lowered.find('select ')
    from_start = lowered.find(' from ')
    if select_start == -1 or from_start == -1:
        return ()
    select_clause = normalized[select_start + 7:from_start]
    result: List[str] = []
    for part in _split_sql_list(select_clause):
        alias = _extract_alias(part)
        result.append(alias or _default_result_label(part))
    return tuple(item.lower() for item in result)


def _parse_query_tail(sql: str, canonical_outputs: Sequence[str], result_columns: Sequence[str]) -> tuple[tuple[Dict[str, Any], ...], int | None]:
    lowered = sql.lower()
    order_idx = lowered.find(' order by ')
    limit_idx = lowered.find(' limit ')
    order_by: List[Dict[str, Any]] = []
    limit: int | None = None
    if limit_idx != -1:
        limit_text = sql[limit_idx + 7:].strip().rstrip(';')
        match = re.match(r'(\d+)', limit_text)
        if match:
            limit = int(match.group(1))
    if order_idx != -1:
        end = limit_idx if limit_idx != -1 and limit_idx > order_idx else len(sql)
        clause = sql[order_idx + 10:end].strip()
        canonical_lookup = {item: result_columns[idx] for idx, item in enumerate(canonical_outputs)}
        for item in _split_sql_list(clause):
            match = re.match(r'(.+?)\s+(asc|desc)\s*$', item, re.IGNORECASE)
            expr = match.group(1).strip() if match else item.strip()
            direction = (match.group(2).lower() if match else 'asc')
            if expr.isdigit():
                idx = int(expr) - 1
                canonical = canonical_outputs[idx] if 0 <= idx < len(canonical_outputs) else expr
                result_label = result_columns[idx] if 0 <= idx < len(result_columns) else expr
            else:
                normalized_expr = _default_result_label(expr)
                canonical = _canonicalize_measure_name(expr)
                if normalized_expr not in result_columns and canonical in canonical_lookup:
                    result_label = canonical_lookup[canonical]
                else:
                    result_label = normalized_expr
            order_by.append({
                'expression': expr.lower(),
                'result_label': result_label,
                'canonical_output': canonical,
                'direction': direction,
            })
    return tuple(order_by), limit


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
    result_columns: tuple[str, ...]
    canonical_output_columns: tuple[str, ...]
    group_by: tuple[str, ...]
    filter_tokens: tuple[str, ...]
    filter_predicates: tuple[str, ...]
    canonical_filter_predicates: tuple[str, ...]
    measure_columns: tuple[str, ...]
    aggregate_functions: tuple[str, ...]
    order_by: tuple[Dict[str, Any], ...]
    limit: int | None
    plan_features: tuple[str, ...]
    description: str
    consumer_surface: Optional[str] = None
    latency_tier: Optional[str] = None
    freshness_tier: Optional[str] = None
    reuse_group: Optional[str] = None
    report_variant_type: Optional[str] = None

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
            "result_columns": list(self.result_columns),
            "canonical_output_columns": list(self.canonical_output_columns),
            "group_by": list(self.group_by),
            "filter_tokens": list(self.filter_tokens),
            "measure_columns": list(self.measure_columns),
            "aggregate_functions": list(self.aggregate_functions),
            "order_by": [dict(item) for item in self.order_by],
            "limit": self.limit,
            "plan_features": list(self.plan_features),
            "description": self.description,
            "consumer_surface": self.consumer_surface,
            "latency_tier": self.latency_tier,
            "freshness_tier": self.freshness_tier,
            "reuse_group": self.reuse_group,
            "report_variant_type": self.report_variant_type,
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
    sla_tier: Optional[str] = None
    business_owner: Optional[str] = None
    dashboard_family: Optional[str] = None
    refresh_mode: Optional[str] = None
    reuse_pressure: Optional[str] = None

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
            "sla_tier": self.sla_tier,
            "business_owner": self.business_owner,
            "dashboard_family": self.dashboard_family,
            "refresh_mode": self.refresh_mode,
            "reuse_pressure": self.reuse_pressure,
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
    task_story: Optional[str] = None
    primary_user: Optional[str] = None
    decision_cycle: Optional[str] = None
    freshness_profile: Optional[str] = None

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
            "task_story": self.task_story,
            "primary_user": self.primary_user,
            "decision_cycle": self.decision_cycle,
            "freshness_profile": self.freshness_profile,
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
                "task_story": self.task_story,
                "primary_user": self.primary_user,
                "decision_cycle": self.decision_cycle,
                "freshness_profile": self.freshness_profile,
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


def _resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        if path.exists():
            return str(path)
        path_parts = path.parts
        for runtime_root in _COMMON_RUNTIME_ROOTS:
            root_parts = Path(runtime_root).parts
            if path_parts[: len(root_parts)] != root_parts:
                continue
            candidate = (_REPO_ROOT / Path(*path_parts[len(root_parts):])).resolve()
            if candidate.exists():
                return str(candidate)
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def _is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        if path.stat().st_size > 512:
            return False
        return path.read_bytes().startswith(_GIT_LFS_POINTER_PREFIX)
    except OSError:
        return False


def _repo_relative_asset_path(path: Path) -> Optional[str]:
    try:
        return path.resolve().relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def resolve_runtime_asset_path(path_str: str) -> str:
    resolved = Path(_resolve_repo_path(path_str))
    if resolved.exists() and not _is_git_lfs_pointer(resolved):
        return str(resolved)

    repo_relative_path = _repo_relative_asset_path(resolved)
    if repo_relative_path is None:
        if resolved.exists():
            raise RuntimeError(f"Task asset is a Git LFS pointer outside repo root: {resolved}")
        raise FileNotFoundError(f"Task asset not found: {resolved}")

    repo_id = os.getenv("HF_SPACE_REPO_ID") or os.getenv("SPACE_ID")
    if not repo_id:
        if resolved.exists():
            raise RuntimeError(
                f"Task asset at {resolved} is a Git LFS pointer. "
                "Set HF_SPACE_REPO_ID or SPACE_ID so the runtime can download the real file."
            )
        raise FileNotFoundError(
            f"Task asset not found at {resolved}. "
            "Set HF_SPACE_REPO_ID or SPACE_ID so the runtime can download it from the Space repo."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to materialize Git LFS-backed task assets at runtime"
        ) from exc

    downloaded_path = hf_hub_download(repo_id=repo_id, repo_type="space", filename=repo_relative_path)
    return str(Path(downloaded_path).resolve())


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
    group_by = tuple(column.lower() for column in payload.get("group_by", []))
    columns = tuple(column.lower() for column in payload.get("columns", []))
    sql_result_columns = _result_columns_from_sql(sql)
    raw_result_columns = payload.get("result_columns")
    if raw_result_columns:
        result_columns = tuple(str(column).lower() for column in raw_result_columns)
    elif len(sql_result_columns) == len(columns):
        result_columns = sql_result_columns
    else:
        result_columns = tuple(_default_result_label(column) for column in payload.get("columns", []))
    raw_canonical_outputs = payload.get("canonical_output_columns") or list(result_columns)
    canonical_output_columns = tuple(
        str(column).lower() if idx < len(group_by) else _canonicalize_measure_name(column)
        for idx, column in enumerate(raw_canonical_outputs)
    )
    order_by = tuple(dict(item) for item in payload.get("order_by", []))
    limit = int(payload["limit"]) if payload.get("limit") is not None else None
    if not order_by and limit is None:
        order_by, limit = _parse_query_tail(sql, canonical_output_columns, result_columns)
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
        columns=columns,
        result_columns=result_columns,
        canonical_output_columns=canonical_output_columns,
        group_by=group_by,
        filter_tokens=tuple(token.lower() for token in payload.get("filter_tokens", [])),
        filter_predicates=filter_predicates,
        canonical_filter_predicates=canonical_filter_predicates,
        measure_columns=tuple(canonical_output_columns[len(group_by):]),
        aggregate_functions=tuple(function.lower() for function in payload.get("aggregate_functions", [])),
        order_by=order_by,
        limit=limit,
        plan_features=tuple(feature.lower() for feature in payload.get("plan_features", [])),
        description=payload.get("description", payload["query_id"]),
        consumer_surface=payload.get("consumer_surface"),
        latency_tier=payload.get("latency_tier"),
        freshness_tier=payload.get("freshness_tier"),
        reuse_group=payload.get("reuse_group"),
        report_variant_type=payload.get("report_variant_type"),
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
        sla_tier=payload.get("sla_tier"),
        business_owner=payload.get("business_owner"),
        dashboard_family=payload.get("dashboard_family"),
        refresh_mode=payload.get("refresh_mode"),
        reuse_pressure=payload.get("reuse_pressure"),
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
                sla_tier=cluster.sla_tier,
                business_owner=cluster.business_owner,
                dashboard_family=cluster.dashboard_family,
                refresh_mode=cluster.refresh_mode,
                reuse_pressure=cluster.reuse_pressure,
            )
        )
    return tuple(enriched)


def load_task_manifest(task_manifest_path: str | Path) -> TaskSpec:
    path = Path(task_manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset_dir = _resolve_repo_path(payload["dataset_dir"])
    database_path = _resolve_repo_path(payload["database_path"])
    tables = tuple(_load_table(item) for item in payload.get("tables", []))
    visible_queries = tuple(_load_query(item) for item in payload.get("visible_queries", []))
    holdout_queries = tuple(_load_query(item) for item in payload.get("holdout_queries", []))
    clusters = _enrich_clusters(tuple(_load_cluster(item) for item in payload.get("clusters", [])), visible_queries)
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
        task_story=payload.get("task_story"),
        primary_user=payload.get("primary_user"),
        decision_cycle=payload.get("decision_cycle"),
        freshness_profile=payload.get("freshness_profile"),
    )


def discover_task_manifests(task_asset_roots: str | Path | Sequence[str | Path] = _TASK_ASSET_ROOTS) -> Dict[str, TaskSpec]:
    if isinstance(task_asset_roots, (str, Path)):
        roots = [Path(task_asset_roots)]
    else:
        roots = [Path(root) for root in task_asset_roots]

    tasks: Dict[str, TaskSpec] = {}
    for root in roots:
        if not root.exists():
            continue
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
