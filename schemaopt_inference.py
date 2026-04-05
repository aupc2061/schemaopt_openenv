"""Inference runner for the standalone schema optimization benchmark.

Environment variables:
- OPENAI_API_KEY: required for OpenAI client auth
- HF_TOKEN: optional fallback token if OPENAI_API_KEY is not set
- MODEL_NAME: optional model id to call (default: gpt-5.4-mini)
- API_BASE_URL: optional custom OpenAI-compatible base URL
- MAX_STEPS: optional max steps per episode (default: 8)
- TASK_ID: optional benchmark task id (default: schemaopt_hard_mobile_revenue_ops)
- MAX_ACTION_RETRIES: optional max model retries per environment step (default: 4)
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemaopt_env.models import SchemaOptAction
from schemaopt_env.server.schemaopt_environment import SchemaOptEnvironment

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4")
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
DEFAULT_MAX_STEPS = int(os.getenv("MAX_STEPS", "25"))
DEFAULT_TASK_ID = os.getenv("TASK_ID", "schemaopt_hard_mobile_revenue_ops")
DEFAULT_MAX_ACTION_RETRIES = int(os.getenv("MAX_ACTION_RETRIES", "4"))

SYSTEM_PROMPT = """You are operating a workload-adaptive schema optimization environment.

Your job is to improve weighted workload cost by creating derived logical objects while preserving semantics.
You cannot rewrite workload queries. Query text is fixed. You must use environment actions only.

Important strategy:
1. Inspect hotspot clusters first.
2. Retrieve query summaries and full context for a small hotspot subset.
3. Copy the target query's predicates, tables, dimensions, and measures exactly unless router diagnostics show a concrete mismatch to fix.
4. Create or modify one focused derived object, then benchmark or inspect router status before making another schema change.
5. Submit once routed improvement is positive, or when repeated router checks show no routeable progress, or when step budget is low.

Operation guide:
- inspect_cluster: inspect one cluster profile and representative query hints using target_id.
- inspect_query: inspect one visible query summary using target_id.
- inspect_query_plan: show baseline vs current routed plan summary for one visible query using target_id.
- inspect_router_status: show route decision, top candidate, and rejection reasons for selected visible queries. Always pass cluster_id or query_ids; never call it unscoped.
- retrieve_queries: retrieve visible query summaries by cluster/pattern/table/column/plan features.
- get_query_context: return full context for specific query_ids, including SQL, exact predicates, canonical predicates, and rewrite template hints.
- create_derived_object: create one derived object from SQL and metadata.
- modify_derived_object: replace an existing derived object definition.
- drop_derived_object: remove one derived object by target_id.
- list_derived_objects: list all derived objects currently available.
- checkpoint: save current derived object set.
- revert_checkpoint: restore to a previous checkpoint.
- benchmark_subset: benchmark specific visible query_ids.
- benchmark_cluster: benchmark all visible queries in one cluster_id.
- submit: run final visible+holdout evaluation and finish episode.

Decision triggers:
- Before the first create_derived_object for a cluster, call inspect_query_plan at least once for that cluster.
- After create_derived_object or modify_derived_object, the next schema-related action must be inspect_router_status or benchmark_cluster/benchmark_subset, scoped to the same intended cluster or explicit query_ids.
- If inspect_router_status shows only predicate_mismatch for the current object, the next schema-related action must be modify_derived_object or drop_derived_object.
- If two consecutive benchmarks show routed_query_count == 0, either submit or explicitly switch clusters based on hotspot evidence.
- If step budget remaining is <= 3, strongly prefer submit.
- Prefer modify_derived_object over creating near-duplicate objects in the same cluster.
- Do not request the same get_query_context for the same query_ids repeatedly unless new router evidence justifies it.
- For inspect_router_status, always include cluster_id or query_ids. Unscoped router checks mix unrelated clusters and are low value.

Action schema:
{
  "operation": "inspect_catalog" | "inspect_table_stats" | "inspect_cluster" | "inspect_query" |
               "inspect_query_plan" | "inspect_router_status" | "retrieve_queries" |
               "get_query_context" | "create_derived_object" | "modify_derived_object" |
               "drop_derived_object" | "list_derived_objects" | "checkpoint" |
               "revert_checkpoint" | "benchmark_subset" | "benchmark_cluster" | "submit",
  "target_id": "required for inspect_table_stats|inspect_cluster|inspect_query|inspect_query_plan",
  "query_ids": ["optional query ids"],
  "pattern": "optional regex or substring",
  "cluster_id": "used by retrieve_queries and required by benchmark_cluster",
  "tables": ["optional tables"],
  "columns": ["optional columns"],
  "plan_features": ["optional plan features"],
  "top_k": 1,
  "object_kind": "optional join_matview|agg_matview|filtered_projection|denorm_table",
  "name": "optional object name",
  "sql_definition": "optional SQL definition",
  "source_objects": ["required for create/modify"],
  "grain_hint": "optional comma-separated grain columns",
  "intended_clusters": ["optional cluster ids"],
  "routing_tags": ["optional routing tags"]
}

Rules:
- Return ONLY a JSON object, with no markdown and no explanation.
- Do not invent query ids or cluster ids; use only ids provided in the observation.
- For inspect_cluster, pass the cluster identifier in target_id (not cluster_id).
- Derived object names must be valid SQL identifiers matching ^[A-Za-z_][A-Za-z0-9_]*$.
- Never use dots, spaces, or schema prefixes in the name field.
- For create_derived_object and modify_derived_object, always include source_objects and ensure they match tables used in sql_definition.
- Never rely on any local heuristic policy. If earlier attempts were invalid, fix the output format and return a valid action.
"""


def _extract_json_object_candidates(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    candidates: List[Dict[str, Any]] = []

    code_fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if code_fence_match:
        fenced = code_fence_match.group(1)
        try:
            payload = json.loads(fenced)
            if isinstance(payload, dict):
                candidates.append(payload)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start == -1:
            break
        try:
            payload, offset = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(payload, dict):
            candidates.append(payload)
        cursor = start + offset
    return candidates


def _normalize_action_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    op = normalized.get("operation")

    inspect_ops = {"inspect_table_stats", "inspect_cluster", "inspect_query", "inspect_query_plan"}
    if op in inspect_ops and not normalized.get("target_id"):
        if normalized.get("cluster_id"):
            normalized["target_id"] = normalized["cluster_id"]
        elif normalized.get("query_id"):
            normalized["target_id"] = normalized["query_id"]

    if op == "benchmark_cluster" and not normalized.get("cluster_id") and normalized.get("target_id"):
        normalized["cluster_id"] = normalized["target_id"]

    return normalized


def parse_action(response_text: str) -> Optional[SchemaOptAction]:
    candidates = _extract_json_object_candidates(response_text)
    if not candidates:
        return None

    for payload in candidates:
        normalized = _normalize_action_payload(payload)
        try:
            return SchemaOptAction(**normalized)
        except Exception as exc:
            print(f"Rejected candidate: {json.dumps(normalized, ensure_ascii=True)} | reason: {exc}")

    return None


def build_user_prompt(
    observation: Any,
    history: List[str],
    step: int,
    benchmark_history: List[Dict[str, Any]],
    cluster_attempts: Dict[str, int],
    query_context_requests: Dict[str, int],
    max_steps: int,
    parse_errors: Optional[List[str]] = None,
) -> str:
    metadata = observation.metadata or {}
    task = metadata.get("task", {})
    history_text = "\n".join(history[-6:]) if history else "(none)"
    budgets = task.get("budgets", {})
    task_max_steps = int(budgets.get("max_steps") or max_steps)
    derived_objects = (observation.catalog_summary or {}).get("derived_objects", [])
    repeated_context_requests = {key: value for key, value in query_context_requests.items() if value > 1}
    prompt_payload = {
        "task": task,
        "catalog_summary": observation.catalog_summary,
        "workload_summary": observation.workload_summary,
        "retrieval_context": observation.retrieval_context,
        "benchmark_context": observation.benchmark_context,
        "router_summary": getattr(observation, "router_summary", {}) or {},
        "action_feedback": observation.action_feedback,
        "recent_history": history_text,
        "recent_benchmark_history": benchmark_history[-2:],
        "cluster_attempts": dict(cluster_attempts),
        "derived_object_summaries": derived_objects,
        "repeated_query_context_requests": repeated_context_requests,
        "remaining_budget_summary": {
            "steps_remaining": max(0, min(max_steps, task_max_steps) - step + 1),
            "max_steps": task_max_steps,
            "max_new_derived_objects": budgets.get("max_new_derived_objects"),
            "max_storage_bytes": budgets.get("max_storage_bytes"),
            "max_refresh_runtime_ms": budgets.get("max_refresh_runtime_ms"),
            "current_derived_object_count": len(derived_objects),
            "current_routed_query_count": (observation.benchmark_context or {}).get("routed_query_count"),
        },
        "step_reward": observation.reward,
        "done": observation.done,
    }

    prompt = json.dumps(prompt_payload, indent=2, default=str)
    if parse_errors:
        prompt += (
            "\n\nPrevious action attempts were invalid. Return only a valid JSON action matching the schema."
            "\nValidation issues:\n- " + "\n- ".join(parse_errors[-3:])
        )
    return prompt


def request_model_action(user_content: str, model_name: str, api_base_url: Optional[str]) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Install it to run model-driven inference.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY or HF_TOKEN must be set for model-driven inference.")

    try:
        client_kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
        if api_base_url:
            client_kwargs["base_url"] = api_base_url
        client = OpenAI(**client_kwargs)
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "developer", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return response.output_text or ""
    except Exception as exc:
        raise RuntimeError(f"Model request failed: {exc}") from exc


def choose_action(
    observation: Any,
    history: List[str],
    step: int,
    benchmark_history: List[Dict[str, Any]],
    cluster_attempts: Dict[str, int],
    query_context_requests: Dict[str, int],
    model_name: str,
    api_base_url: Optional[str],
    max_steps: int,
    max_action_retries: int,
) -> SchemaOptAction:
    errors: List[str] = []
    for attempt in range(1, max_action_retries + 1):
        user_content = build_user_prompt(
            observation,
            history,
            step,
            benchmark_history,
            cluster_attempts,
            query_context_requests,
            max_steps,
            parse_errors=errors,
        )
        response_text = request_model_action(user_content, model_name, api_base_url)
        if not response_text.strip():
            errors.append("empty model response")
            continue
        action = parse_action(response_text)
        if action is not None:
            print(f"Model action parsed successfully on attempt {attempt}.")
            return action
        truncated = response_text.strip().replace("\n", " ")
        errors.append(f"unparseable action payload: {truncated[:300]}")

    raise RuntimeError(
        f"Model failed to produce a valid SchemaOptAction after {max_action_retries} attempts."
    )


def run_episode(
    task_id: str,
    model_name: str,
    api_base_url: Optional[str],
    max_steps: int,
    max_action_retries: int,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    env = SchemaOptEnvironment()
    observation = env.reset(task_id=task_id)
    history: List[str] = []
    benchmark_history: List[Dict[str, Any]] = []
    cluster_attempts: Counter[str] = Counter()
    query_context_requests: Counter[str] = Counter()
    total_reward = 0.0

    print(f"Starting schema optimization inference run for {task_id}")

    terminated_due_to_max_steps = False

    for step in range(1, max_steps + 1):
        action = choose_action(
            observation,
            history,
            step,
            benchmark_history,
            dict(cluster_attempts),
            dict(query_context_requests),
            model_name,
            api_base_url,
            max_steps,
            max_action_retries,
        )
        print(f"Step {step}: {json.dumps(action.model_dump(exclude_none=True), ensure_ascii=True)}")

        observation = env.step(action)
        reward = observation.reward or 0.0
        total_reward += reward

        if action.operation in {"benchmark_cluster", "benchmark_subset"}:
            benchmark_history.append(
                {
                    "operation": action.operation,
                    "cluster_id": action.cluster_id,
                    "query_ids": list(action.query_ids),
                    "benchmark_context": observation.benchmark_context,
                    "router_summary": getattr(observation, "router_summary", {}) or {},
                }
            )
        if action.operation in {"create_derived_object", "modify_derived_object"}:
            cluster_key = ",".join(action.intended_clusters) if action.intended_clusters else "(unscoped)"
            cluster_attempts[cluster_key] += 1
        if action.operation == "get_query_context":
            query_context_requests["|".join(sorted(action.query_ids))] += 1

        history_line = (
            f"Step {step}: {action.operation} -> reward {reward:+.3f}, "
            f"status={observation.status}, done={observation.done}, "
            f"router={json.dumps(getattr(observation, 'router_summary', {}) or {}, ensure_ascii=True)}"
        )
        history.append(history_line)

        print(f"  Reward: {reward:+.3f} | Done: {observation.done} | Message: {observation.message}")

        if observation.done:
            print("Episode complete.")
            break
    else:
        terminated_due_to_max_steps = True
        print(f"Reached max steps ({max_steps}) without a model-issued submit action.")

    final_feedback = observation.action_feedback or {}
    result = {
        "task_id": task_id,
        "final_score": env.state.final_score,
        "total_reward": round(total_reward, 6),
        "steps": env.state.step_count,
        "derived_object_count": env.state.derived_object_count,
        "benchmark_runs": env.state.benchmark_runs,
        "final_message": observation.message,
        "terminated_due_to_max_steps": terminated_due_to_max_steps,
        "done": observation.done,
        "benchmark_context": observation.benchmark_context,
        "router_summary": getattr(observation, "router_summary", {}) or {},
        "final_feedback": final_feedback,
    }

    print("\nFinal result:")
    result_path = output_path or Path(f"inference_result_{task_id}_{model_name}.json")
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-driven inference against schemaopt_env.")
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID, help="Task id to run.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model id for the OpenAI-compatible API.")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL, help="Optional OpenAI-compatible API base URL.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum number of environment steps.")
    parser.add_argument("--max-action-retries", type=int, default=DEFAULT_MAX_ACTION_RETRIES, help="Maximum model retries per step.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for the result JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_episode(
        task_id=args.task_id,
        model_name=args.model_name,
        api_base_url=args.api_base_url,
        max_steps=args.max_steps,
        max_action_retries=args.max_action_retries,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
