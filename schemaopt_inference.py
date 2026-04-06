"""Inference runner for the standalone schema optimization benchmark.

Environment variables:
- OPENAI_API_KEY: required for OpenAI client auth
- HF_TOKEN: optional fallback token if OPENAI_API_KEY is not set
- MODEL_NAME: optional model id to call (default: gpt-5.4-mini)
- API_BASE_URL: optional custom OpenAI-compatible base URL
- MAX_STEPS: optional max steps per episode (defaults to the task budget when unset)
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

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4-mini")
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
DEFAULT_MAX_STEPS = os.getenv("MAX_STEPS")
DEFAULT_TASK_ID = os.getenv("TASK_ID", "schemaopt_hard_mobile_revenue_ops")
DEFAULT_MAX_ACTION_RETRIES = int(os.getenv("MAX_ACTION_RETRIES", "4"))

SYSTEM_PROMPT = """You are operating a workload-adaptive schema optimization environment.

Your job is to maximize final whole-workload score, not to stop after the first local win.
The final objective depends on visible and holdout gated improvement, correctness coverage, migration quality, and storage efficiency.
You cannot rewrite workload queries directly. Query text is fixed. You must use environment actions only.

Optimization objective:
- Prefer actions that improve the overall visible+holdout workload, not just one cluster in isolation.
- A cluster benchmark is local evidence only. Do not assume a strong cluster-local gain means the whole workload is solved.
- Favor multiple correct, reusable objects across high-hotspot clusters over one narrow object with early submit.
- Correctness is mandatory. A wider but incorrect routing pattern is worse than a narrower correct one.

Primary loop:
1. Call get_cluster_context for a hotspot or currently unsolved cluster.
2. Create or modify one focused derived object for that cluster.
3. Immediately call inspect_rewrite_status or benchmark_cluster/benchmark_subset for the same scope.
4. Use decision_state to decide whether to continue exploring other hotspots or submit.
5. Submit only when overall progress is likely near-best for the remaining step budget.

Operation guide:
- inspect_catalog: inspect base and derived objects.
- inspect_table_stats: inspect one base table using target_id.
- get_cluster_context: return one cluster summary, representative visible queries, query context, router summary, latest benchmark snapshot, and relevant derived objects. Always pass cluster_id.
- inspect_rewrite_status: inspect plan and routing status for exactly one scope: target_id, cluster_id, or query_ids.
- create_derived_object: create one derived object from SQL and metadata.
- modify_derived_object: replace an existing derived object definition.
- drop_derived_object: remove one derived object by target_id.
- benchmark_subset: benchmark specific visible query_ids.
- benchmark_cluster: benchmark all visible queries in one cluster_id.
- submit: run final visible+holdout evaluation and finish episode.

Decision policy:
- Before the first create_derived_object for a cluster, call get_cluster_context for that cluster.
- After create_derived_object or modify_derived_object, the next schema-related action should be inspect_rewrite_status or benchmark_cluster/benchmark_subset for the same scope.
- Prefer modify_derived_object over creating near-duplicate objects in the same cluster.
- Treat decision_state.best_visible_gated_improvement as the best local verified gain seen so far, not the final workload gain.
- If only one cluster is verified_positive and several hotspot clusters remain untouched, prefer exploring the next unsolved hotspot instead of submitting.
- On hard tasks, avoid immediate submit after a single verified-positive cluster unless remaining steps are very low or other top hotspots have already been checked and look unpromising.
- Prefer covering additional top-hotspot clusters when the current object is narrow and decision_state.unsolved_clusters is still large.
- Strongly prefer submit only when one of these is true:
  - decision_state.phase == "submit" and there is evidence of more than one useful object or more than one resolved hotspot,
  - remaining_budget_summary.steps_remaining <= 3 and there is at least one verified_positive cluster,
  - most top-hotspot clusters have been attempted and further exploration is unlikely to beat the current solution.
- Do not repeat get_cluster_context for the same cluster unless rewrite or benchmark evidence changed your plan.
- Do not submit immediately after one positive benchmark if the task is hard and the remaining budget is still ample.

Action schema:
{
  "operation": "inspect_catalog" | "inspect_table_stats" | "get_cluster_context" |
               "inspect_rewrite_status" | "create_derived_object" |
               "modify_derived_object" | "drop_derived_object" |
               "benchmark_subset" | "benchmark_cluster" | "submit",
  "target_id": "required for inspect_table_stats and optional single-query inspect_rewrite_status",
  "query_ids": ["optional visible query ids"],
  "cluster_id": "required for get_cluster_context and benchmark_cluster; optional for inspect_rewrite_status",
  "tables": ["optional tables"],
  "columns": ["optional columns"],
  "plan_features": ["optional plan features"],
  "top_k": 2,
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
- inspect_rewrite_status must include exactly one scope: target_id, cluster_id, or query_ids.
- Derived object names must match ^[A-Za-z_][A-Za-z0-9_]*$.
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

    if op == "get_cluster_context" and not normalized.get("cluster_id") and normalized.get("target_id"):
        normalized["cluster_id"] = normalized["target_id"]
    if op == "benchmark_cluster" and not normalized.get("cluster_id") and normalized.get("target_id"):
        normalized["cluster_id"] = normalized["target_id"]
    if op == "inspect_rewrite_status":
        if normalized.get("query_id") and not normalized.get("query_ids"):
            normalized["query_ids"] = [normalized["query_id"]]
        if normalized.get("target_id") and normalized.get("cluster_id"):
            normalized.pop("cluster_id", None)
        if normalized.get("target_id") and normalized.get("query_ids"):
            normalized.pop("query_ids", None)
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
    cluster_context_requests: Dict[str, int],
    max_steps: int,
    parse_errors: Optional[List[str]] = None,
) -> str:
    metadata = observation.metadata or {}
    task = metadata.get("task", {})
    budgets = task.get("budgets", {})
    history_text = "\n".join(history[-6:]) if history else "(none)"
    decision_state = getattr(observation, "decision_state", {}) or {}
    repeated_cluster_context_requests = {
        key: value for key, value in cluster_context_requests.items() if value > 1
    }
    derived_object_summaries = (observation.catalog_summary or {}).get("derived_objects", [])
    prompt_payload = {
        "task": {
            "task_id": task.get("task_id"),
            "difficulty": task.get("difficulty"),
            "budgets": budgets,
        },
        "decision_state": decision_state,
        "benchmark_context": observation.benchmark_context,
        "router_summary": getattr(observation, "router_summary", {}) or {},
        "action_feedback": observation.action_feedback,
        "recent_history": history_text,
        "recent_benchmark_history": benchmark_history[-3:],
        "cluster_attempts": dict(cluster_attempts),
        "repeated_cluster_context_requests": repeated_cluster_context_requests,
        "derived_object_summaries": derived_object_summaries,
        "compact_fallbacks": {
            "task_budgets": budgets,
            "workload_clusters": (observation.workload_summary or {}).get("clusters", []),
        },
        "remaining_budget_summary": decision_state.get("remaining_budget_summary", {
            "steps_remaining": max(0, max_steps - step + 1),
            "max_steps": max_steps,
        }),
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
    cluster_context_requests: Dict[str, int],
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
            cluster_context_requests,
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
    max_steps: Optional[int],
    max_action_retries: int,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    env = SchemaOptEnvironment()
    observation = env.reset(task_id=task_id)
    budgets = (observation.metadata or {}).get("task", {}).get("budgets", {})
    task_budget = budgets.get("max_steps")
    if max_steps is None:
        effective_max_steps = int(task_budget or 0)
    elif task_budget is None:
        effective_max_steps = max_steps
    else:
        effective_max_steps = min(max_steps, int(task_budget))
    history: List[str] = []
    benchmark_history: List[Dict[str, Any]] = []
    cluster_attempts: Counter[str] = Counter()
    cluster_context_requests: Counter[str] = Counter()
    total_reward = 0.0

    print(f"Starting schema optimization inference run for {task_id}")

    terminated_due_to_max_steps = False

    for step in range(1, effective_max_steps + 1):
        action = choose_action(
            observation,
            history,
            step,
            benchmark_history,
            dict(cluster_attempts),
            dict(cluster_context_requests),
            model_name,
            api_base_url,
            effective_max_steps,
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
                    "decision_state": getattr(observation, "decision_state", {}) or {},
                }
            )
        if action.operation in {"create_derived_object", "modify_derived_object"}:
            for cluster_id in action.intended_clusters:
                cluster_attempts[cluster_id] += 1
        if action.operation == "get_cluster_context" and action.cluster_id:
            cluster_context_requests[action.cluster_id] += 1

        history_line = (
            f"Step {step}: {action.operation} -> reward {reward:+.3f}, "
            f"status={observation.status}, done={observation.done}, "
            f"decision={json.dumps(getattr(observation, 'decision_state', {}) or {}, ensure_ascii=True)}"
        )
        history.append(history_line)

        print(f"  Reward: {reward:+.3f} | Done: {observation.done} | Message: {observation.message}")

        if observation.done:
            print("Episode complete.")
            break
    else:
        terminated_due_to_max_steps = True
        print(f"Reached max steps ({effective_max_steps}) without a model-issued submit action.")

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
        "decision_state": getattr(observation, "decision_state", {}) or {},
        "final_feedback": final_feedback,
    }

    print("\nFinal result:")
    result_path = output_path or Path(f"inference_result_{task_id}_{model_name}_v2.json")
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-driven inference against schemaopt_env.")
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID, help="Task id to run.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model id for the OpenAI-compatible API.")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL, help="Optional OpenAI-compatible API base URL.")
    parser.add_argument("--max-steps", type=int, default=int(DEFAULT_MAX_STEPS) if DEFAULT_MAX_STEPS else None, help="Maximum number of environment steps. Defaults to the task budget when omitted.")
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
