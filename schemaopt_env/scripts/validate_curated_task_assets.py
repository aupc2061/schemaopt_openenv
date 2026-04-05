from __future__ import annotations

import json
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPO_ROOT / "schemaopt_env" / "task_assets"
TASK_IDS = [
    "schemaopt_easy_hiring_pipeline",
    "schemaopt_easy_product_adoption",
    "schemaopt_medium_campaign_performance",
    "schemaopt_medium_delivery_operations",
    "schemaopt_hard_lifecycle_engagement",
    "schemaopt_hard_mobile_revenue_ops",
]


def main() -> None:
    failures: list[tuple[str, str, str]] = []
    for task_id in TASK_IDS:
        manifest_path = ASSET_ROOT / f"{task_id}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        db_path = REPO_ROOT / str(manifest["database_path"])
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            for section in ("visible_queries", "holdout_queries"):
                for payload in manifest.get(section, []):
                    query_id = str(payload["query_id"])
                    sql = str(payload["sql"])
                    try:
                        con.execute(f"SELECT * FROM ({sql}) AS q LIMIT 0")
                    except Exception as exc:  # noqa: BLE001
                        failures.append((task_id, query_id, str(exc)))
        finally:
            con.close()

    if failures:
        for task_id, query_id, error in failures:
            print(f"FAIL {task_id} {query_id}: {error}")
        raise SystemExit(1)
    print(f"validated {len(TASK_IDS)} curated tasks with no query compilation errors")


if __name__ == "__main__":
    main()
