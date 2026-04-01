"""Build manifest-backed schema optimization task assets from dacomp-de-impl datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "datasets" / "dacomp-de"
ASSET_ROOT = REPO_ROOT / "schemaopt_env" / "task_assets"

SELECTED_TASKS = [
    ("schemaopt_easy_lever", "easy", "lever", 1),
    ("schemaopt_easy_pendo", "easy", "pendo", 2),
    ("schemaopt_easy_customer360", "easy", "customer360", 3),
    ("schemaopt_easy_salesforce", "easy", "salesforce", 4),
    ("schemaopt_medium_google_ads", "medium", "google_ads", 5),
    ("schemaopt_medium_ad_reporting", "medium", "ad_reporting", 6),
    ("schemaopt_medium_app_reporting", "medium", "app_reporting", 7),
    ("schemaopt_medium_asana", "medium", "asana", 8),
    ("schemaopt_hard_google_play", "hard", "google_play", 9),
    ("schemaopt_hard_hubspot", "hard", "hubspot", 10),
    ("schemaopt_hard_jira", "hard", "jira", 11),
    ("schemaopt_hard_marketo", "hard", "marketo", 12),
]

DIFFICULTY_CONFIG: Dict[str, Dict[str, Any]] = {
    "easy": {
        "cluster_count": 3,
        "visible_per_cluster": 4,
        "holdout_per_cluster": 2,
        "budgets": {
            "max_new_derived_objects": 3,
            "max_storage_bytes": 12_000_000,
            "max_refresh_runtime_ms": 8000.0,
            "max_steps": 18,
        },
    },
    "medium": {
        "cluster_count": 4,
        "visible_per_cluster": 6,
        "holdout_per_cluster": 3,
        "budgets": {
            "max_new_derived_objects": 4,
            "max_storage_bytes": 24_000_000,
            "max_refresh_runtime_ms": 15000.0,
            "max_steps": 24,
        },
    },
    "hard": {
        "cluster_count": 5,
        "visible_per_cluster": 7,
        "holdout_per_cluster": 4,
        "budgets": {
            "max_new_derived_objects": 5,
            "max_storage_bytes": 40_000_000,
            "max_refresh_runtime_ms": 25000.0,
            "max_steps": 30,
        },
    },
}

PREFERRED_OBJECTS = ["agg_matview", "denorm_table", "agg_matview", "join_matview", "agg_matview"]


@dataclass
class QueryShape:
    cluster_index: int
    label: str
    business_label: str
    preferred_object_kind: str
    sql: str
    tables: List[str]
    columns: List[str]
    group_by: List[str]
    filter_predicates: List[str]
    measure_columns: List[str]
    aggregate_functions: List[str]
    plan_features: List[str]
    description: str


def _parse_simple_schema_yaml(path: Path) -> Dict[str, Dict[str, str]]:
    schema: Dict[str, Dict[str, str]] = {}
    current_table: Optional[str] = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0 and line.endswith(":"):
            current_table = line[:-1]
            schema[current_table] = {}
            continue
        if indent >= 2 and current_table and ":" in line:
            key, value = line.split(":", 1)
            schema[current_table][key.strip()] = value.strip()
    return schema


def _table_columns(schema: Dict[str, Dict[str, str]], table: str) -> Dict[str, str]:
    return schema.get(table, {})


def _is_time_column(column: str, dtype: str) -> bool:
    column_lc = column.lower()
    dtype_lc = dtype.lower()
    return (
        "date" in column_lc
        or column_lc.endswith("_at")
        or "time" in column_lc
        or "date" in dtype_lc
        or "timestamp" in dtype_lc
    )


def _is_numeric(dtype: str) -> bool:
    dtype_lc = dtype.lower()
    return any(token in dtype_lc for token in ["int", "double", "decimal", "numeric", "real", "float"])


def _is_text(dtype: str) -> bool:
    dtype_lc = dtype.lower()
    return any(token in dtype_lc for token in ["char", "text", "string", "varchar", "uuid"])


def _pick_time_column(columns: Dict[str, str]) -> Optional[str]:
    for name, dtype in columns.items():
        if _is_time_column(name, dtype):
            return name
    return None


def _pick_dimension_columns(columns: Dict[str, str], excluded: Iterable[str]) -> List[str]:
    blocked = {value.lower() for value in excluded}
    dims: List[str] = []
    for name, dtype in columns.items():
        lower = name.lower()
        if lower in blocked or lower == "id" or lower.endswith("_id"):
            continue
        if _is_text(dtype) or dtype.lower() == "boolean":
            dims.append(name)
    return dims


def _pick_measure_column(columns: Dict[str, str], excluded: Iterable[str]) -> Optional[str]:
    blocked = {value.lower() for value in excluded}
    for name, dtype in columns.items():
        lower = name.lower()
        if lower in blocked:
            continue
        if _is_numeric(dtype) and lower not in {"id"} and not lower.endswith("_id"):
            return name
    return None


def _find_join_target(base_table: str, schema: Dict[str, Dict[str, str]]) -> Optional[tuple[str, str]]:
    base_columns = schema.get(base_table, {})
    for column in base_columns:
        if not column.endswith("_id") or column.lower() == "id":
            continue
        target_name = column[:-3]
        if target_name in schema and "id" in schema[target_name]:
            return target_name, column
        for table_name, target_columns in schema.items():
            if table_name == base_table:
                continue
            if table_name.endswith(target_name) and "id" in target_columns:
                return table_name, column
    return None


def _quoted(identifier: str) -> str:
    return f'"{identifier}"'


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().lower().split())


def _build_query_shapes(domain: str, difficulty: str, schema: Dict[str, Dict[str, str]]) -> List[QueryShape]:
    config = DIFFICULTY_CONFIG[difficulty]
    candidate_tables = sorted(
        schema,
        key=lambda table: (
            0 if _pick_time_column(schema[table]) else 1,
            -len(schema[table]),
            table,
        ),
    )
    selected = candidate_tables[: config["cluster_count"]]
    shapes: List[QueryShape] = []

    for cluster_index, base_table in enumerate(selected, start=1):
        base_cols = _table_columns(schema, base_table)
        time_col = _pick_time_column(base_cols)
        join_target = _find_join_target(base_table, schema)
        join_table = join_target[0] if join_target else None
        join_key = join_target[1] if join_target else None
        join_cols = _table_columns(schema, join_table) if join_table else {}

        dims: List[tuple[str, str]] = []
        filters: List[str] = []
        tables = [f"raw.{base_table}"]
        from_sql = f"raw.{base_table} b"
        plan_features: List[str] = ["aggregate", "filter"]

        if time_col:
            dims.append(("metric_period", f"date_trunc('month', try_cast(b.{_quoted(time_col)} AS TIMESTAMP))"))
            filters.append(f"try_cast(b.{_quoted(time_col)} AS TIMESTAMP) IS NOT NULL")
            plan_features.append("time_bucket")

        base_dim_candidates = _pick_dimension_columns(base_cols, excluded=[time_col or ""])[:2]
        if base_dim_candidates:
            first_dim = base_dim_candidates[0]
            dims.append((first_dim.lower(), f"b.{_quoted(first_dim)}"))
            filters.append(f"b.{_quoted(first_dim)} IS NOT NULL")

        if join_table and join_key:
            join_dim_candidates = _pick_dimension_columns(join_cols, excluded=[])[:1]
            if join_dim_candidates:
                join_dim = join_dim_candidates[0]
                dims.append((f"{join_table.lower()}_{join_dim.lower()}", f"j.{_quoted(join_dim)}"))
                tables.append(f"raw.{join_table}")
                from_sql += f" JOIN raw.{join_table} j ON b.{_quoted(join_key)} = j.id"
                filters.append(f"j.{_quoted(join_dim)} IS NOT NULL")
                plan_features.append("join")

        measure_col = _pick_measure_column(base_cols, excluded=[time_col or "", *(dim for dim, _ in dims)])
        measures = [("row_count", "COUNT(*)", "count")]
        if measure_col:
            measures.append((f"sum_{measure_col.lower()}", f"SUM(b.{_quoted(measure_col)})", "sum"))
            plan_features.append("numeric_aggregation")

        group_variants: List[List[tuple[str, str]]] = []
        if len(dims) >= 3:
            group_variants.append(dims[:3])
        if len(dims) >= 2:
            group_variants.append(dims[:2])
        if len(dims) >= 1:
            group_variants.append(dims[:1])
        if not group_variants:
            group_variants.append([])
        if len(group_variants) == 1:
            group_variants.append(group_variants[0])

        group_variants = group_variants[:3]
        preferred = PREFERRED_OBJECTS[(cluster_index - 1) % len(PREFERRED_OBJECTS)]
        label = f"{base_table} workload"
        business_label = f"{domain} {base_table} analytics"

        for variant_index, selected_dims in enumerate(group_variants, start=1):
            select_exprs = [f"{expr} AS {alias}" for alias, expr in selected_dims]
            measure_exprs = [f"{expr} AS {alias}" for alias, expr, _ in measures]
            select_sql = ", ".join(select_exprs + measure_exprs)
            sql = f"SELECT {select_sql} FROM {from_sql}"
            if filters:
                sql += " WHERE " + " AND ".join(filters)
            if selected_dims:
                sql += " GROUP BY " + ", ".join(alias for alias, _ in selected_dims)
            shape = QueryShape(
                cluster_index=cluster_index,
                label=label,
                business_label=business_label,
                preferred_object_kind=preferred,
                sql=sql,
                tables=tables,
                columns=[alias for alias, _ in selected_dims] + [alias for alias, _, _ in measures],
                group_by=[alias for alias, _ in selected_dims],
                filter_predicates=filters,
                measure_columns=[alias for alias, _, _ in measures],
                aggregate_functions=[kind for _, _, kind in measures],
                plan_features=sorted(set(plan_features + (["wide_group"] if len(selected_dims) >= 2 else []))),
                description=f"{business_label} variant {variant_index}",
            )
            shapes.append(shape)

    return shapes


def _expand_cluster_queries(task_id: str, difficulty: str, shapes: Sequence[QueryShape]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    config = DIFFICULTY_CONFIG[difficulty]
    visible_queries: List[Dict[str, Any]] = []
    holdout_queries: List[Dict[str, Any]] = []
    clusters: List[Dict[str, Any]] = []

    for cluster_index in sorted({shape.cluster_index for shape in shapes}):
        cluster_shapes = [shape for shape in shapes if shape.cluster_index == cluster_index]
        cluster_id = f"{task_id}_cluster_{cluster_index:02d}"
        visible_count = config["visible_per_cluster"]
        holdout_count = config["holdout_per_cluster"]
        generated_ids: List[str] = []
        total_frequency = 0.0
        total_weight = 0.0

        for i in range(visible_count):
            shape = cluster_shapes[i % len(cluster_shapes)]
            query_id = f"{task_id}_vq_c{cluster_index:02d}_{i + 1:02d}"
            frequency_weight = round(1.0 + cluster_index * 0.45 + (i % 3) * 0.15, 2)
            priority_weight = round(1.0 + (i % 2) * 0.10, 2)
            payload = {
                "query_id": query_id,
                "sql": shape.sql,
                "normalized_sql": _normalize_sql(shape.sql),
                "cluster_id": cluster_id,
                "business_tag": shape.business_label,
                "frequency_weight": frequency_weight,
                "priority_weight": priority_weight,
                "tables": shape.tables,
                "columns": shape.columns,
                "group_by": shape.group_by,
                "filter_tokens": [predicate.lower().replace(" ", "_") for predicate in shape.filter_predicates],
                "filter_predicates": shape.filter_predicates,
                "measure_columns": shape.measure_columns,
                "aggregate_functions": shape.aggregate_functions,
                "plan_features": shape.plan_features,
                "description": f"{shape.description} visible {i + 1}",
            }
            visible_queries.append(payload)
            generated_ids.append(query_id)
            total_frequency += frequency_weight
            total_weight += frequency_weight * priority_weight

        for i in range(holdout_count):
            shape = cluster_shapes[(i + 1) % len(cluster_shapes)]
            query_id = f"{task_id}_hq_c{cluster_index:02d}_{i + 1:02d}"
            frequency_weight = round(0.8 + cluster_index * 0.30 + (i % 2) * 0.10, 2)
            priority_weight = round(1.0 + ((i + 1) % 2) * 0.05, 2)
            holdout_queries.append(
                {
                    "query_id": query_id,
                    "sql": shape.sql,
                    "normalized_sql": _normalize_sql(shape.sql),
                    "cluster_id": cluster_id,
                    "business_tag": shape.business_label,
                    "frequency_weight": frequency_weight,
                    "priority_weight": priority_weight,
                    "tables": shape.tables,
                    "columns": shape.columns,
                    "group_by": shape.group_by,
                    "filter_tokens": [predicate.lower().replace(" ", "_") for predicate in shape.filter_predicates],
                    "filter_predicates": shape.filter_predicates,
                    "measure_columns": shape.measure_columns,
                    "aggregate_functions": shape.aggregate_functions,
                    "plan_features": shape.plan_features,
                    "description": f"{shape.description} holdout {i + 1}",
                }
            )

        representative = cluster_shapes[0]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "label": representative.label,
                "business_label": representative.business_label,
                "query_ids": generated_ids,
                "query_count": len(generated_ids),
                "total_frequency_weight": round(total_frequency, 4),
                "total_weighted_baseline_cost": round(total_weight, 4),
                "top_tables": representative.tables,
                "common_operator_patterns": representative.plan_features,
                "representative_dimensions": representative.group_by,
                "representative_measures": representative.measure_columns,
                "hotspot_rank": cluster_index,
                "preferred_object_kind": representative.preferred_object_kind,
            }
        )

    clusters.sort(key=lambda item: item["total_weighted_baseline_cost"], reverse=True)
    for rank, cluster in enumerate(clusters, start=1):
        cluster["hotspot_rank"] = rank
    return visible_queries, holdout_queries, clusters


def _tables_payload(schema: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    payload = []
    for table_name, columns in schema.items():
        payload.append(
            {
                "name": f"raw.{table_name}",
                "columns": [{"name": column, "type": dtype} for column, dtype in columns.items()],
                "row_count": 0,
            }
        )
    return payload


def build_manifest(task_id: str, difficulty: str, domain: str, dataset_index: int) -> Dict[str, Any]:
    dataset_dir = DATASET_ROOT / f"dacomp-de-impl-{dataset_index:03d}"
    start_yaml = next(dataset_dir.glob("*_start.yaml"))
    db_file = next(dataset_dir.glob("*_start.duckdb"))
    schema = _parse_simple_schema_yaml(start_yaml)
    shapes = _build_query_shapes(domain, difficulty, schema)
    visible_queries, holdout_queries, clusters = _expand_cluster_queries(task_id, difficulty, shapes)
    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "domain": domain,
        "objective": f"Optimize the {domain.replace('_', ' ')} workload over the real DuckDB source database by materializing derived objects that reduce measured execution cost while preserving exact query results.",
        "seed_source": str(dataset_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "dataset_dir": str(dataset_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "database_path": str(db_file.relative_to(REPO_ROOT)).replace("\\", "/"),
        "tables": _tables_payload(schema),
        "visible_queries": visible_queries,
        "holdout_queries": holdout_queries,
        "clusters": clusters,
        "budgets": DIFFICULTY_CONFIG[difficulty]["budgets"],
        "allowed_object_kinds": ["join_matview", "agg_matview", "filtered_projection", "denorm_table"],
        "engine_capabilities": {
            "engine": "duckdb",
            "uses_real_duckdb_database": True,
            "uses_real_query_execution": True,
            "uses_explain_plan_metrics": True,
            "rewrite_model": "explicit_rewritten_sql",
        },
    }


def main() -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    for task_id, difficulty, domain, dataset_index in SELECTED_TASKS:
        payload = build_manifest(task_id, difficulty, domain, dataset_index)
        output_path = ASSET_ROOT / f"{task_id}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
