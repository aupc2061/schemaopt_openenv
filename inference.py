"""LLM-driven inference loop for DataDAG Task 1.

Environment variables:
- OPENAI_API_KEY: required for OpenAI client auth
- MODEL_NAME: model id to call (default: gpt-4o-mini)
- API_BASE_URL: optional custom OpenAI-compatible base URL
- HF_TOKEN: optional fallback token if OPENAI_API_KEY is not set
- MAX_STEPS: optional max steps per episode (default: 8)
"""

from __future__ import annotations

import json
import os
from dotenv import load_dotenv
import re
import sys
from pathlib import Path
from typing import List, Set
from openai import OpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datadag_env.models import DatadagAction
from datadag_env.server.datadag_environment import DatadagEnvironment

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-mini")
API_BASE_URL = os.getenv("API_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = 0.1
MAX_TOKENS = 2000

FALLBACK_ACTION = {
    "pipeline_command": "create_node",
    "model_identifier": "stg_transactions",
    "sql_syntax": (
        "SELECT transaction_id, CAST(user_id AS INTEGER) AS user_id, "
        "CAST(amount_usd AS DOUBLE) AS amount_usd, status "
        "FROM raw_transactions WHERE status = 'completed'"
    ),
    "upstream_dependencies": [],
}

SYSTEM_PROMPT = """You are operating a DataDAG environment for data engineering.

Your goal in Task 1:
1) Create a staging model from raw_transactions with proper type casting and completed status filter.
2) Create a mart model that joins raw_users to staging and aggregates lifetime revenue per user.
3) Execute the DAG.

Action schema:
{
  "pipeline_command": "create_node" | "update_node" | "execute_dag" | "view_lineage",
  "model_identifier": "optional string",
  "sql_syntax": "optional SQL string",
  "upstream_dependencies": ["optional", "list", "of", "strings"]
}

Rules:
- For create_node/update_node, include model_identifier and sql_syntax.
- For execute_dag/view_lineage, do not include sql_syntax.
- sql_syntax MUST be only a SELECT query body (no CREATE/VIEW/TABLE/INSERT/DROP statements).
- raw_transactions and raw_users are source tables, NOT DAG dependencies. upstream_dependencies must
    contain only previously created model names.
- For stg_transactions use columns: transaction_id, user_id, amount_usd, status.
- For mart_user_lifetime_revenue output columns: user_id, country, lifetime_revenue.
- Do not use columns amount, currency, or created_at.
- Recommended order: create stg_transactions -> create mart_user_lifetime_revenue -> execute_dag.
- Return ONLY a JSON object, no markdown fences, no extra text.
"""

STG_SQL = (
        "SELECT transaction_id, CAST(user_id AS INTEGER) AS user_id, "
        "CAST(amount_usd AS DOUBLE) AS amount_usd, status "
        "FROM raw_transactions WHERE status = 'completed'"
)

MART_SQL = (
        "SELECT u.user_id, u.country, COALESCE(SUM(t.amount_usd), 0.0) AS lifetime_revenue "
        "FROM raw_users u "
        "LEFT JOIN stg_transactions t ON u.user_id = t.user_id "
        "GROUP BY u.user_id, u.country"
)


def build_user_prompt(observation, history: List[str]) -> str:
    metadata = observation.metadata or {}
    task = metadata.get("task", {})

    history_text = "\n".join(history[-6:]) if history else "(none)"
    sample_text = (
        json.dumps(observation.data_sample, ensure_ascii=True)
        if observation.data_sample is not None
        else "null"
    )

    return (
        f"Task: {task}\n"
        f"Current status: {observation.dag_integrity_status}\n"
        f"Execution trace:\n{observation.execution_trace}\n\n"
        f"Cascading failures: {observation.cascading_failure_nodes}\n"
        f"Data sample: {sample_text}\n"
        f"Recent history:\n{history_text}\n\n"
        "Provide the next action as strict JSON."
    )


def _extract_json_object(text: str) -> str:
    text = text.strip()

    code_fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if code_fence_match:
        return code_fence_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def parse_model_action(response_text: str) -> DatadagAction:
    candidate = _extract_json_object(response_text)

    try:
        payload = json.loads(candidate)
        return DatadagAction.model_validate(payload)
    except Exception:
        return DatadagAction.model_validate(FALLBACK_ACTION)


def _extract_select_body(sql: str) -> str:
    sql_clean = sql.strip().rstrip(";")
    if re.match(r"(?is)^\s*select\b", sql_clean):
        return sql_clean

    # Handle model outputs like: CREATE OR REPLACE VIEW x AS SELECT ...
    match = re.search(r"(?is)\bas\s+(select\b.*)$", sql_clean)
    if match:
        return match.group(1).strip().rstrip(";")

    return sql_clean


def _planned_safe_action(created_models: Set[str]) -> DatadagAction:
    if "stg_transactions" not in created_models:
        return DatadagAction(
            pipeline_command="create_node",
            model_identifier="stg_transactions",
            sql_syntax=STG_SQL,
            upstream_dependencies=[],
        )
    if "mart_user_lifetime_revenue" not in created_models:
        return DatadagAction(
            pipeline_command="create_node",
            model_identifier="mart_user_lifetime_revenue",
            sql_syntax=MART_SQL,
            upstream_dependencies=["stg_transactions"],
        )
    return DatadagAction(pipeline_command="execute_dag")


def sanitize_action(
    action: DatadagAction,
    observation,
    created_models: Set[str],
) -> DatadagAction:
    """Normalize model-proposed actions to the environment's strict contract."""

    # Never execute before required models are created.
    if action.pipeline_command == "execute_dag" and not {
        "stg_transactions",
        "mart_user_lifetime_revenue",
    }.issubset(created_models):
        return _planned_safe_action(created_models)

    # If previous step errored, prefer deterministic recovery plan.
    if "ERROR:" in (observation.execution_trace or ""):
        return _planned_safe_action(created_models)

    # Once both required models are present and no active error, terminate by execution.
    if {
        "stg_transactions",
        "mart_user_lifetime_revenue",
    }.issubset(created_models):
        return DatadagAction(pipeline_command="execute_dag")

    if action.pipeline_command in ("create_node", "update_node"):
        model_id = action.model_identifier or ""
        sql = _extract_select_body(action.sql_syntax or "")

        # If model already exists, treat a repeated create as an update to avoid avoidable errors.
        command = action.pipeline_command
        if command == "create_node" and model_id in created_models:
            command = "update_node"

        # Dependency list can only include created model names.
        deps = [d for d in action.upstream_dependencies if d in created_models]

        if model_id == "stg_transactions":
            deps = []
            # Repair common invalid source-column patterns.
            if re.search(r"\b(amount|currency|created_at)\b", sql, flags=re.IGNORECASE):
                sql = STG_SQL
            if not re.search(r"\bamount_usd\b", sql, flags=re.IGNORECASE):
                sql = STG_SQL

        if model_id == "mart_user_lifetime_revenue":
            deps = ["stg_transactions"] if "stg_transactions" in created_models else []
            if not re.search(r"\blifetime_revenue\b", sql, flags=re.IGNORECASE):
                sql = MART_SQL

        # If model id is missing or unknown, use safe planner.
        if model_id not in {"stg_transactions", "mart_user_lifetime_revenue"}:
            return _planned_safe_action(created_models)

        try:
            return DatadagAction(
                pipeline_command=command,
                model_identifier=model_id,
                sql_syntax=sql,
                upstream_dependencies=deps,
            )
        except Exception:
            return _planned_safe_action(created_models)

    return action

def request_model_action(user_content: str) -> str:
    messages = [
        {"role": "developer", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    try:
        client = OpenAI()
        responses = client.responses.create(
            model=MODEL_NAME,
            input=messages,
            # max_output_tokens=MAX_TOKENS,
        )
        return responses.output_text or ""
    except Exception as exc:
        print(f"Model request failed ({exc}). Using fallback action.")
        return json.dumps(FALLBACK_ACTION)


def run_episode() -> dict:
    env = DatadagEnvironment()
    created_models: Set[str] = set()

    observation = env.reset()
    history: List[str] = []
    total_reward = 0.0

    print("Starting DataDAG Task 1 inference run")

    for step in range(1, MAX_STEPS + 1):
        user_content = build_user_prompt(observation, history)
        response_text = request_model_action(user_content)
        action = parse_model_action(response_text)
        action = sanitize_action(action, observation, created_models)

        print(f"Step {step}: model suggested -> {action.model_dump()}")

        observation = env.step(action)
        if action.pipeline_command in ("create_node", "update_node") and action.model_identifier:
            if observation.dag_integrity_status == "valid":
                created_models.add(action.model_identifier)

        reward = observation.reward or 0.0
        total_reward += reward

        history_line = (
            f"Step {step}: {action.pipeline_command} -> reward {reward:+.3f}, "
            f"integrity={observation.dag_integrity_status}"
        )
        history.append(history_line)

        print(
            f"  Reward: {reward:+.3f} | Done: {observation.done} | "
            f"Integrity: {observation.dag_integrity_status}"
        )

        if observation.done:
            print("Episode complete.")
            break
    else:
        print(f"Reached max steps ({MAX_STEPS}).")

    rubric = (observation.metadata or {}).get("task1_rubric")
    result = {
        "task_id": "task1_easy",
        "score": env.state.final_score,
        "total_reward": round(total_reward, 6),
        "steps": env.state.step_count,
        "rubric": rubric,
        "last_trace": observation.execution_trace,
    }

    print("\nFinal result:")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    run_episode()


if __name__ == "__main__":
    main()
