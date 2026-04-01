"""Inference runner for the standalone schema optimization benchmark.

Environment variables:
- OPENAI_API_KEY: required for OpenAI client auth
- HF_TOKEN: optional fallback token if OPENAI_API_KEY is not set
- MODEL_NAME: optional model id to call (default: gpt-5.4-mini)
- API_BASE_URL: optional custom OpenAI-compatible base URL
- MAX_STEPS: optional max steps per episode (default: 8)
- TASK_ID: optional benchmark task id (default: schemaopt_hard_google_play)
- MAX_ACTION_RETRIES: optional max model retries per environment step (default: 4)
"""

from __future__ import annotations

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

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4")
API_BASE_URL = os.getenv("API_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
TASK_ID = os.getenv("TASK_ID", "schemaopt_hard_google_play")
MAX_ACTION_RETRIES = int(os.getenv("MAX_ACTION_RETRIES", "4"))

SYSTEM_PROMPT = """You are operating a workload-adaptive schema optimization environment.

Your job is to improve weighted workload cost by creating derived logical objects while preserving semantics.
You cannot rewrite workload queries. Query text is fixed. You must use environment actions only.

Important strategy:
1. Inspect hotspot clusters first.
2. Retrieve query summaries and full context for a small hotspot subset.
3. Create one focused derived object that matches the hotspot cluster's tables, dimensions, and filters.
4. Benchmark a cluster or subset before creating more objects.
5. Submit once you have a positive gated improvement or step budget is low.

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
  "source_objects": ["optional source objects"],
  "grain_hint": "optional comma-separated grain columns",
  "intended_clusters": ["optional cluster ids"],
  "routing_tags": ["optional routing tags"]
}

Rules:
- Return ONLY a JSON object, with no markdown and no explanation.
- Do not invent query ids or cluster ids; use only ids provided in the observation.
- For inspect_cluster, pass the cluster identifier in target_id (not cluster_id).
- Prefer benchmarking before submitting.
- Derived object names must be valid SQL identifiers matching ^[A-Za-z_][A-Za-z0-9_]*$.
- Never use dots, spaces, or schema prefixes in the `name` field. Use names like `agg_installs_monthly`.
- Never rely on any local heuristic policy. If earlier attempts were invalid, fix the output format and return a valid action.
"""


def _extract_json_object_candidates(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    # print(f"Extracting JSON object from model response: {text[:500]}")
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

    if candidates:
        # print(f"Extracted {len(candidates)} JSON object candidate(s) from text.")
        for payload in candidates[:3]:
            # print(f"  Candidate: {json.dumps(payload, ensure_ascii=True)}")
            pass

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


def build_user_prompt(observation: Any, history: List[str], parse_errors: Optional[List[str]] = None) -> str:
    metadata = observation.metadata or {}
    task = metadata.get("task", {})
    history_text = "\n".join(history[-6:]) if history else "(none)"

    prompt_payload = {
        "task": task,
        "catalog_summary": observation.catalog_summary,
        "workload_summary": observation.workload_summary,
        "retrieval_context": observation.retrieval_context,
        "benchmark_context": observation.benchmark_context,
        "action_feedback": observation.action_feedback,
        "recent_history": history_text,
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


def request_model_action(user_content: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Install it to run model-driven inference.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY or HF_TOKEN must be set for model-driven inference.")

    try:
        client_kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
        if API_BASE_URL:
            client_kwargs["base_url"] = API_BASE_URL
        client = OpenAI(**client_kwargs)
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "developer", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return response.output_text or ""
    except Exception as exc:
        raise RuntimeError(f"Model request failed: {exc}") from exc


def choose_action(observation: Any, history: List[str], step: int) -> SchemaOptAction:
    errors: List[str] = []
    for attempt in range(1, MAX_ACTION_RETRIES + 1):
        user_content = build_user_prompt(observation, history, parse_errors=errors)
        response_text = request_model_action(user_content)
        # print(f"Response text: {response_text}")
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
        f"Model failed to produce a valid SchemaOptAction after {MAX_ACTION_RETRIES} attempts."
    )


def run_episode(task_id: str = TASK_ID) -> Dict[str, Any]:
    env = SchemaOptEnvironment()
    observation = env.reset(task_id=task_id)
    history: List[str] = []
    total_reward = 0.0

    print(f"Starting schema optimization inference run for {task_id}")

    terminated_due_to_max_steps = False

    for step in range(1, MAX_STEPS + 1):
        action = choose_action(observation, history, step)
        print(f"Step {step}: {json.dumps(action.model_dump(exclude_none=True), ensure_ascii=True)}")

        observation = env.step(action)
        reward = observation.reward or 0.0
        total_reward += reward

        history_line = (
            f"Step {step}: {action.operation} -> reward {reward:+.3f}, "
            f"status={observation.status}, done={observation.done}"
        )
        history.append(history_line)

        print(f"  Reward: {reward:+.3f} | Done: {observation.done} | Message: {observation.message}")

        if observation.done:
            print("Episode complete.")
            break
    else:
        terminated_due_to_max_steps = True
        print(f"Reached max steps ({MAX_STEPS}) without a model-issued submit action.")

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
        "final_feedback": final_feedback,
    }

    print("\nFinal result:")
    with open(f"inference_result_{task_id}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    run_episode()


if __name__ == "__main__":
    main()

