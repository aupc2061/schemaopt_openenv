"""Backfill explicit output-shape metadata into curated DACOMP task manifests."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from schemaopt_env.tasks import (
    _canonicalize_measure_name,
    _default_result_label,
    _normalize_sql,
    _parse_query_tail,
    _result_columns_from_sql,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPO_ROOT / "schemaopt_env" / "task_assets"
CURATED_TASK_IDS = [
    "schemaopt_easy_hiring_pipeline",
    "schemaopt_easy_product_adoption",
    "schemaopt_medium_campaign_performance",
    "schemaopt_medium_delivery_operations",
    "schemaopt_hard_lifecycle_engagement",
    "schemaopt_hard_mobile_revenue_ops",
]


def _rewrite_group_by_ordinals(sql: str, group_by_count: int) -> str:
    if group_by_count <= 0:
        return sql
    pattern = re.compile(r"(?is)\bgroup\s+by\b(.*?)(\border\s+by\b|\blimit\b|$)")
    replacement = " GROUP BY " + ", ".join(str(index) for index in range(1, group_by_count + 1))

    def _replace(match: re.Match[str]) -> str:
        suffix = match.group(2) or ""
        if suffix:
            return replacement + " " + suffix
        return replacement

    return pattern.sub(_replace, sql, count=1)


def _enrich_query_payload(payload: dict[str, Any]) -> bool:
    sql = str(payload.get("sql", "")).strip()
    if not sql:
        return False
    group_by = [str(column).lower() for column in payload.get("group_by", [])]
    normalized_sql = str(payload.get("normalized_sql", "")).strip()
    if group_by:
        sql = _rewrite_group_by_ordinals(sql, len(group_by))
        normalized_sql = _normalize_sql(sql)
    elif not normalized_sql:
        normalized_sql = _normalize_sql(sql)

    result_columns = list(_result_columns_from_sql(sql))
    if not result_columns:
        result_columns = [_default_result_label(column) for column in payload.get("columns", [])]
    canonical_output_columns = [
        result_columns[idx] if idx < len(group_by) else _canonicalize_measure_name(result_columns[idx])
        for idx in range(len(result_columns))
    ]
    order_by, limit = _parse_query_tail(sql, canonical_output_columns, result_columns)

    changed = False
    if payload.get("sql") != sql:
        payload["sql"] = sql
        changed = True
    if payload.get("normalized_sql") != normalized_sql:
        payload["normalized_sql"] = normalized_sql
        changed = True
    if payload.get("result_columns") != result_columns:
        payload["result_columns"] = result_columns
        changed = True
    if payload.get("canonical_output_columns") != canonical_output_columns:
        payload["canonical_output_columns"] = canonical_output_columns
        changed = True
    if payload.get("measure_columns") != canonical_output_columns[len(group_by):]:
        payload["measure_columns"] = canonical_output_columns[len(group_by):]
        changed = True
    order_by_payload = [dict(item) for item in order_by]
    if payload.get("order_by") != order_by_payload or "order_by" not in payload:
        payload["order_by"] = order_by_payload
        changed = True
    if payload.get("limit") != limit or "limit" not in payload:
        payload["limit"] = limit
        changed = True
    return changed


def main() -> None:
    for task_id in CURATED_TASK_IDS:
        path = ASSET_ROOT / f"{task_id}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for section in ("visible_queries", "holdout_queries"):
            for payload in manifest.get(section, []):
                changed = _enrich_query_payload(payload) or changed
        if changed:
            path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"updated {task_id}")
        else:
            print(f"unchanged {task_id}")


if __name__ == "__main__":
    main()

