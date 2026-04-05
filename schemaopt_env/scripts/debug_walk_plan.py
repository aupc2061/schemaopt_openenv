"""Debug helper for inspecting SchemaOptEnvironment._walk_plan recursion.

Usage:
    python -m schemaopt_env.scripts.debug_walk_plan
    python -m schemaopt_env.scripts.debug_walk_plan --task-id schemaopt_easy_hiring_pipeline
    python -m schemaopt_env.scripts.debug_walk_plan --query-id q01
    python -m schemaopt_env.scripts.debug_walk_plan --show-json
"""

from __future__ import annotations

import argparse
import json
from pprint import pprint
from typing import Any, Dict, List, Tuple

from schemaopt_env.models import SchemaOptAction
from schemaopt_env.server.schemaopt_environment import SchemaOptEnvironment, _BLOCKING
from schemaopt_env.tasks import TASK_CATALOG


def _trace_walk(
    env: SchemaOptEnvironment,
    node: Any,
    depth: int,
    path: str,
    rows: List[Dict[str, Any]],
) -> Tuple[int, int, int, int, List[str]]:
    indent = "  " * depth

    if isinstance(node, list):
        print(f"{indent}[LIST] path={path}, len={len(node)}")
        summaries = [
            _trace_walk(env, item, depth + 1, f"{path}[{idx}]", rows)
            for idx, item in enumerate(node)
        ]
        if not summaries:
            result = (1, 0, 0, 0, [])
        else:
            result = (
                max(item[0] for item in summaries),
                sum(item[1] for item in summaries),
                sum(item[2] for item in summaries),
                sum(item[3] for item in summaries),
                [name for item in summaries for name in item[4]],
            )
        print(
            f"{indent}[LIST-SUMMARY] path={path} depth={result[0]} ops={result[1]} joins={result[2]} blocking={result[3]}"
        )
        rows.append(
            {
                "path": path,
                "kind": "list",
                "plan_depth": result[0],
                "operator_count": result[1],
                "join_count": result[2],
                "blocking_operator_count": result[3],
                "operators": result[4],
            }
        )
        return result

    if isinstance(node, dict):
        name = str(
            node.get("name")
            or node.get("operator_name")
            or node.get("operator_type")
            or node.get("type")
            or ""
        ).upper()
        raw_children = node.get("children") or node.get("child") or node.get("plans") or []
        if isinstance(raw_children, dict):
            children = [raw_children]
        elif isinstance(raw_children, list):
            children = raw_children
        else:
            children = []

        print(
            f"{indent}[DICT] path={path} name={name or '<none>'} children={len(children)} keys={sorted(str(k) for k in node.keys())}"
        )
        child = (
            _trace_walk(env, children, depth + 1, f"{path}.children", rows)
            if children
            else (0, 0, 0, 0, [])
        )
        result = (
            max(1, child[0] + 1 if children else 1),
            child[1] + (1 if name else 0),
            child[2] + (1 if "JOIN" in name else 0),
            child[3] + (1 if name in _BLOCKING or "AGGREGATE" in name else 0),
            ([name] if name else []) + child[4],
        )
        print(
            f"{indent}[DICT-SUMMARY] path={path} depth={result[0]} ops={result[1]} joins={result[2]} blocking={result[3]}"
        )
        rows.append(
            {
                "path": path,
                "kind": "dict",
                "name": name,
                "child_count": len(children),
                "plan_depth": result[0],
                "operator_count": result[1],
                "join_count": result[2],
                "blocking_operator_count": result[3],
                "operators": result[4],
            }
        )
        return result

    print(f"{indent}[OTHER] path={path} type={type(node).__name__} value={str(node)[:120]}")
    result = (1, 0, 0, 0, [])
    rows.append(
        {
            "path": path,
            "kind": type(node).__name__,
            "value": str(node)[:200],
            "plan_depth": result[0],
            "operator_count": result[1],
            "join_count": result[2],
            "blocking_operator_count": result[3],
            "operators": result[4],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug _walk_plan traversal for one query")
    parser.add_argument("--task-id", help="Task id to load", default=None)
    parser.add_argument("--query-id", help="Visible query id to inspect", default=None)
    parser.add_argument("--show-json", action="store_true", help="Print raw EXPLAIN JSON too")
    args = parser.parse_args()

    task_id = args.task_id or next(iter(TASK_CATALOG.keys()))
    env = SchemaOptEnvironment()
    env.reset(task_id=task_id)
    if env._task is None:
        raise RuntimeError("Environment task is not initialized after reset")

    query_id = args.query_id or env._task.visible_queries[0].query_id
    query = env._get_visible_query(query_id)
    print("SQL query:", query)

    print(f"Task: {task_id}")
    print(f"Query: {query_id}")

    baseline = env._baseline_for_query(query)
    print("\nOriginal plan summary:")
    pprint(baseline.plan.summary())

    if baseline.plan.raw_explain_json is None:
        print("\nNo JSON plan available from EXPLAIN (FORMAT json).")
        return

    if args.show_json:
        print("\nRaw EXPLAIN JSON:")
        print(json.dumps(baseline.plan.raw_explain_json, indent=2))

    trace_rows: List[Dict[str, Any]] = []

    original_walk_plan = env._walk_plan

    def patched_walk_plan(node: Any) -> Tuple[int, int, int, int, List[str]]:
        return _trace_walk(env, node=node, depth=0, path="root", rows=trace_rows)

    env._walk_plan = patched_walk_plan  # type: ignore[assignment]
    try:
        print("\nTracing _walk_plan recursion:")
        traced = env._walk_plan(baseline.plan.raw_explain_json)
        print("\nFinal traced tuple:")
        pprint(
            {
                "plan_depth": traced[0],
                "operator_count": traced[1],
                "join_count": traced[2],
                "blocking_operator_count": traced[3],
                "operators": traced[4],
            }
        )
    finally:
        env._walk_plan = original_walk_plan  # type: ignore[assignment]

    print("\nSample trace rows (first 10):")
    pprint(trace_rows[:10])


if __name__ == "__main__":
    main()
