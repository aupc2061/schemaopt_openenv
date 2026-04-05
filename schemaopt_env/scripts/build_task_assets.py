"""Build base SchemaOpt task assets from the local benchmark datasets.

Note:
This script produces seed manifests only. The checked-in curated task assets may
include hand-authored metadata, adjusted visible/holdout mixes, and count tuning
that are applied after generation. Do not rerun this script on the curated suite
unless you intend to refresh those manual edits as well.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "datasets" / "schemaopt_sources"
ASSET_ROOT = REPO_ROOT / "schemaopt_env" / "task_assets"

SELECTED_TASKS = [
    ("schemaopt_easy_hiring_pipeline", "easy", "lever", 1),
    ("schemaopt_easy_product_adoption", "easy", "pendo", 2),
    ("schemaopt_medium_campaign_performance", "medium", "google_ads", 5),
    ("schemaopt_medium_delivery_operations", "medium", "asana", 8),
    ("schemaopt_hard_mobile_revenue_ops", "hard", "google_play", 9),
    ("schemaopt_hard_lifecycle_engagement", "hard", "marketo", 12),
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


def _time_parse_expression(column: str, dtype: str, alias: str = "b") -> str:
    column_ref = f"{alias}.{_quoted(column)}"
    dtype_lc = dtype.lower()
    if any(token in dtype_lc for token in ["char", "text", "string", "varchar"]):
        return (
            f"coalesce("
            f"try_strptime({column_ref}, '%Y-%m-%d %H:%M:%S.%f %z'), "
            f"try_strptime({column_ref}, '%Y-%m-%d %H:%M:%S %z'), "
            f"try_strptime({column_ref}, '%Y-%m-%d %H:%M:%S.%f'), "
            f"try_strptime({column_ref}, '%Y-%m-%d %H:%M:%S'), "
            f"try_cast({column_ref} AS TIMESTAMP)"
            f")"
        )
    return f"try_cast({column_ref} AS TIMESTAMP)"


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


def _canonicalize_measure_name(value: str) -> str:
    normalized = value.strip().lower().replace('\"', '')
    normalized = re.sub(r'\s+', ' ', normalized)
    if normalized in {'count(*)', 'count_star()', 'count_star'}:
        return 'count_star'
    match = re.fullmatch(r'(count|sum|avg|min|max)\((.+)\)', normalized)
    if match:
        func = match.group(1)
        target = match.group(2).strip()
        if target == '*':
            return 'count_star'
        target = target.replace('.', '_')
        target = re.sub(r'[^a-z0-9_]+', '_', target).strip('_')
        return f'{func}_{target}' if target else func
    normalized = normalized.replace('.', '_')
    normalized = re.sub(r'[^a-z0-9_]+', '_', normalized).strip('_')
    return normalized


def _default_result_label(value: str) -> str:
    normalized = value.strip().lower().replace('\"', '')
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


def _parse_query_tail(sql: str, canonical_outputs: Sequence[str], result_columns: Sequence[str]) -> tuple[list[Dict[str, Any]], int | None]:
    lowered = sql.lower()
    order_idx = lowered.find(' order by ')
    limit_idx = lowered.find(' limit ')
    order_by: list[Dict[str, Any]] = []
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
    return order_by, limit


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
            time_expr = _time_parse_expression(time_col, base_cols[time_col])
            dims.append(("metric_period", f"date_trunc('month', {time_expr})"))
            filters.append(f"{time_expr} IS NOT NULL")
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
                sql += " GROUP BY " + ", ".join(str(index) for index in range(1, len(selected_dims) + 1))
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
            result_columns = list(_result_columns_from_sql(shape.sql))
            canonical_output_columns = [
                result_columns[idx] if idx < len(shape.group_by) else _canonicalize_measure_name(result_columns[idx])
                for idx in range(len(result_columns))
            ]
            order_by, limit = _parse_query_tail(shape.sql, canonical_output_columns, result_columns)
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
                "result_columns": result_columns,
                "canonical_output_columns": canonical_output_columns,
                "group_by": shape.group_by,
                "filter_tokens": [predicate.lower().replace(" ", "_") for predicate in shape.filter_predicates],
                "filter_predicates": shape.filter_predicates,
                "measure_columns": shape.measure_columns,
                "aggregate_functions": shape.aggregate_functions,
                "order_by": order_by,
                "limit": limit,
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
            result_columns = list(_result_columns_from_sql(shape.sql))
            canonical_output_columns = [
                result_columns[idx] if idx < len(shape.group_by) else _canonicalize_measure_name(result_columns[idx])
                for idx in range(len(result_columns))
            ]
            order_by, limit = _parse_query_tail(shape.sql, canonical_output_columns, result_columns)
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
                    "result_columns": result_columns,
                    "canonical_output_columns": canonical_output_columns,
                    "group_by": shape.group_by,
                    "filter_tokens": [predicate.lower().replace(" ", "_") for predicate in shape.filter_predicates],
                    "filter_predicates": shape.filter_predicates,
                    "measure_columns": shape.measure_columns,
                    "aggregate_functions": shape.aggregate_functions,
                    "order_by": order_by,
                    "limit": limit,
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
    dataset_dir = DATASET_ROOT / f"source-{dataset_index:03d}"
    start_yaml = next(dataset_dir.glob("*_start.yaml"))
    db_file = next(dataset_dir.glob("*_start.duckdb"))
    schema = _parse_simple_schema_yaml(start_yaml)
    shapes = _build_query_shapes(domain, difficulty, schema)
    visible_queries, holdout_queries, clusters = _expand_cluster_queries(task_id, difficulty, shapes)
    curated_dir = Path("schemaopt_env") / "task_assets" / "schemaopt_curated" / domain
    curated_db = curated_dir / f"{domain}_start.duckdb"
    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "domain": domain,
        "objective": f"Optimize the {domain.replace('_', ' ')} workload over the real DuckDB source database by materializing derived objects that reduce measured execution cost while preserving exact query results.",
        "seed_source": str(curated_dir).replace("\\", "/"),
        "dataset_dir": str(curated_dir).replace("\\", "/"),
        "database_path": str(curated_db).replace("\\", "/"),
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

