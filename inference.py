"""Inference runner for the standalone schema optimization benchmark.

Submission-facing root runner for SchemaOpt. This file contains the full
policy logic and emits structured stdout logs using the required
[START] / [STEP] / [END] tags.

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
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
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

from models import SchemaOptAction
from server.schemaopt_environment import SchemaOptEnvironment


def _safe_int_from_env(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer, got: {raw!r}") from exc


def _sanitize_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", value.strip())
    return sanitized or "unknown"


def _shorten_for_log(text: str, max_len: int = 280) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return f"{compact[: max_len - 3]}..."


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _bool_to_token(value: bool) -> str:
    return "true" if value else "false"


def _fmt_reward(value: float) -> str:
    return f"{float(value):.2f}"


def _fmt_error_token(observation: Any) -> str:
    if getattr(observation, "status", "ok") != "error":
        return "null"
    feedback = getattr(observation, "action_feedback", {}) or {}
    raw_error = feedback.get("error")
    if raw_error is None:
        return "null"
    compact = " ".join(str(raw_error).split())
    return json.dumps(compact, ensure_ascii=True)


def _fmt_action_token(action: SchemaOptAction) -> str:
    payload = action.model_dump(exclude_none=True)
    payload.pop("episode_id", None)
    compact = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return compact.replace(" ", "_")


def _print_start_line(task: str, env_name: str, model: str) -> None:
    print(
        f"[START] task={task} env={env_name} model={model}",
        flush=True,
    )


def _print_step_line(step: int, action: SchemaOptAction, reward: float, done: bool, error_token: str) -> None:
    print(
        f"[STEP] step={step} action={_fmt_action_token(action)} reward={_fmt_reward(reward)} done={_bool_to_token(done)} error={error_token}",
        flush=True,
    )


def _print_end_line(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    reward_list = ",".join(_fmt_reward(item) for item in rewards)
    print(
        f"[END] success={_bool_to_token(success)} steps={steps} score={_fmt_reward(score)} rewards={reward_list}",
        flush=True,
    )


def _task_list_from_arg(raw: str) -> List[str]:
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4-mini")
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
DEFAULT_MAX_STEPS = _safe_int_from_env("MAX_STEPS", None)
DEFAULT_TASK_ID = os.getenv("TASK_ID", "schemaopt_hard_mobile_revenue_ops")
DEFAULT_MAX_ACTION_RETRIES = _safe_int_from_env("MAX_ACTION_RETRIES", 4) or 4

SYSTEM_PROMPT = """You are acting in a schema optimization environment.

Objective:
- Maximize final episode reward using valid environment actions only.
- Query text is fixed. You cannot edit workload queries directly.

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
- Return ONLY a single JSON object, with no markdown and no explanation.
- Use only ids and fields provided by the observation.
- inspect_rewrite_status must include exactly one scope: target_id, cluster_id, or query_ids.
- Derived object names must match ^[A-Za-z_][A-Za-z0-9_]*$.
- Never use dots, spaces, or schema prefixes in the name field.
- For create_derived_object and modify_derived_object, always include source_objects and ensure they match tables used in sql_definition.
- If a previous attempt was invalid, return a valid action that matches the schema.
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
    # Parsing may repair superficial syntax/field aliases, but must not invent
    # scope, ids, or strategy on the model's behalf.
    normalized = dict(payload)
    op = normalized.get("operation")

    if op == "inspect_rewrite_status":
        if normalized.get("query_id") and not normalized.get("query_ids"):
            normalized["query_ids"] = [normalized["query_id"]]
        normalized.pop("query_id", None)
    return normalized


def parse_action(response_text: str) -> tuple[Optional[SchemaOptAction], List[str]]:
    candidates = _extract_json_object_candidates(response_text)
    if not candidates:
        return None, ["no JSON object candidate found"]

    parse_issues: List[str] = []

    for idx, payload in enumerate(candidates, start=1):
        normalized = _normalize_action_payload(payload)
        try:
            return SchemaOptAction(**normalized), parse_issues
        except Exception as exc:
            issue = f"candidate {idx}: {exc}"
            parse_issues.append(issue)
            print(f"Rejected candidate: {json.dumps(normalized, ensure_ascii=True)} | reason: {exc}")

    return None, parse_issues[-5:]


def _prompt_task_view(observation: Any) -> Dict[str, Any]:
    metadata = observation.metadata or {}
    task = metadata.get("task", {})
    return {
        "task_id": task.get("task_id") or task.get("id"),
        "difficulty": task.get("difficulty"),
        "budgets": task.get("budgets", {}),
    }


def _prompt_observation_view(observation: Any) -> Dict[str, Any]:
    decision_state = getattr(observation, "decision_state", {}) or {}
    payload = {
        "task": _prompt_task_view(observation),
        "decision_state": decision_state,
        "benchmark_context": getattr(observation, "benchmark_context", {}) or {},
        "router_summary": getattr(observation, "router_summary", {}) or {},
        "action_feedback": getattr(observation, "action_feedback", {}) or {},
        "done": bool(getattr(observation, "done", False)),
    }
    derived_object_summaries = ((getattr(observation, "catalog_summary", {}) or {}).get("derived_objects")) or []
    if derived_object_summaries:
        payload["derived_object_summaries"] = derived_object_summaries
    return payload


def _prompt_retry_context(parse_errors: Optional[List[str]]) -> str:
    if not parse_errors:
        return ""
    return (
        "\n\nPrevious action attempts were invalid. Return only a valid JSON action matching the schema."
        "\nValidation issues:\n- " + "\n- ".join(parse_errors[-3:])
    )


def build_user_prompt(
    observation: Any,
    history: List[str],
    step: int,
    benchmark_history: List[Dict[str, Any]],
    cluster_context_requests: Dict[str, int],
    max_steps: int,
    parse_errors: Optional[List[str]] = None,
) -> str:
    history_text = "\n".join(history[-6:]) if history else "(none)"
    prompt_payload = _prompt_observation_view(observation)
    prompt_payload["recent_history"] = history_text
    if "remaining_budget_summary" not in prompt_payload["decision_state"]:
        prompt_payload["remaining_budget_summary"] = {
            "steps_remaining": max(0, max_steps - step + 1),
            "max_steps": max_steps,
        }

    prompt = json.dumps(prompt_payload, indent=2, default=str)
    return prompt + _prompt_retry_context(parse_errors)


def request_model_action(
    user_content: str,
    model_name: str,
    api_base_url: Optional[str],
    system_prompt: str,
) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Install it to run model-driven inference.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY or HF_TOKEN must be set for model-driven inference.")

    try:
        client_key = f"{api_base_url or ''}|{OPENAI_API_KEY}"
        if not hasattr(request_model_action, "_clients"):
            request_model_action._clients = {}
        client_cache: Dict[str, Any] = request_model_action._clients
        client = client_cache.get(client_key)
        if client is None:
            client_kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
            if api_base_url:
                client_kwargs["base_url"] = api_base_url
            client = OpenAI(**client_kwargs)
            client_cache[client_key] = client
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "developer", "content": system_prompt},
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
    cluster_context_requests: Dict[str, int],
    model_name: str,
    api_base_url: Optional[str],
    max_steps: int,
    max_action_retries: int,
    system_prompt: str,
) -> SchemaOptAction:
    if max_action_retries < 1:
        raise RuntimeError("max_action_retries must be at least 1")

    errors: List[str] = []
    for attempt in range(1, max_action_retries + 1):
        print(f"Choosing action: step={step} attempt={attempt}/{max_action_retries}")
        user_content = build_user_prompt(
            observation,
            history,
            step,
            benchmark_history,
            cluster_context_requests,
            max_steps,
            parse_errors=errors,
        )
        response_text = request_model_action(user_content, model_name, api_base_url, system_prompt)
        if not response_text.strip():
            errors.append("empty model response")
            print(f"Attempt {attempt} produced an empty model response.")
            continue
        action, parse_issues = parse_action(response_text)
        if action is not None:
            print(f"Model action parsed successfully on attempt {attempt}.")
            return action
        truncated = response_text.strip().replace("\n", " ")
        errors.append(f"unparseable action payload: {truncated[:300]}")
        if parse_issues:
            errors.extend(parse_issues[-2:])
            print(f"Attempt {attempt} parse issues: {_shorten_for_log(' | '.join(parse_issues[-2:]))}")

    tail = " | ".join(errors[-3:]) if errors else "unknown error"
    raise RuntimeError(
        f"Model failed to produce a valid SchemaOptAction after {max_action_retries} attempts. "
        f"Recent issues: {tail}"
    )


def run_episode(
    task_id: str,
    model_name: str,
    api_base_url: Optional[str],
    max_steps: Optional[int],
    max_action_retries: int,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if max_steps is not None and max_steps < 1:
        raise RuntimeError("max_steps must be at least 1 when provided")
    if max_action_retries < 1:
        raise RuntimeError("max_action_retries must be at least 1")

    env = SchemaOptEnvironment()
    observation = None
    budgets: Dict[str, Any] = {}
    final_feedback: Dict[str, Any] = {}
    termination_reason = "runner_exception"
    task_budget: Optional[int] = None
    effective_max_steps = max_steps if max_steps is not None else 0
    if effective_max_steps < 0:
        effective_max_steps = 0
    total_reward = 0.0
    rewards: List[float] = []
    action_trace: List[Dict[str, Any]] = []
    caught_error: Optional[str] = None

    _print_start_line(task_id, "schemaopt_env", model_name)

    terminated_due_to_max_steps = False
    try:
        observation = env.reset(task_id=task_id)
        budgets = (observation.metadata or {}).get("task", {}).get("budgets", {})
        task_budget = budgets.get("max_steps")
        if max_steps is None:
            effective_max_steps = int(task_budget or 0)
        elif task_budget is None:
            effective_max_steps = max_steps
        else:
            effective_max_steps = min(max_steps, int(task_budget))

        if effective_max_steps < 1:
            raise RuntimeError(
                "Effective max steps resolved to 0. Set --max-steps >= 1 or ensure task budget.max_steps is present."
            )

        history: List[str] = []
        benchmark_history: List[Dict[str, Any]] = []
        cluster_context_requests: Counter[str] = Counter()

        for step in range(1, effective_max_steps + 1):
            if observation.done:
                break

            with io.StringIO() as sink, redirect_stdout(sink):
                action = choose_action(
                    observation,
                    history,
                    step,
                    benchmark_history,
                    dict(cluster_context_requests),
                    model_name,
                    api_base_url,
                    effective_max_steps,
                    max_action_retries,
                    SYSTEM_PROMPT,
                )

            observation = env.step(action)
            reward = float(observation.reward or 0.0)
            total_reward += reward
            rewards.append(reward)

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
            if action.operation == "get_cluster_context" and action.cluster_id:
                cluster_context_requests[action.cluster_id] += 1

            history_line = (
                f"Step {step}: {action.operation} -> reward {reward:+.3f}, "
                f"status={observation.status}, done={observation.done}, "
                f"decision={json.dumps(getattr(observation, 'decision_state', {}) or {}, ensure_ascii=True)}"
            )
            history.append(history_line)
            action_trace.append(
                {
                    "step": step,
                    "operation": action.operation,
                    "status": observation.status,
                    "reward": round(float(reward), 6),
                    "done": bool(observation.done),
                    "message": observation.message,
                }
            )

            _print_step_line(
                step=step,
                action=action,
                reward=reward,
                done=bool(observation.done),
                error_token=_fmt_error_token(observation),
            )

            if observation.done:
                break
        else:
            terminated_due_to_max_steps = True
            termination_reason = "runner_max_steps_reached"

    except Exception as exc:
        caught_error = str(exc)
        termination_reason = "runner_exception"
        final_feedback = {"error": caught_error}
        if not rewards:
            rewards = []
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    final_feedback = (observation.action_feedback if observation is not None else None) or final_feedback
    warnings: List[str] = []
    if observation is None or not observation.done:
        warnings.append("episode_not_completed")
    if env.state.final_score is None:
        warnings.append("final_score_unavailable")
    if caught_error is not None:
        warnings.append("runner_exception")

    if observation is not None and observation.done:
        termination_reason = str(final_feedback.get("termination_reason") or "submitted")
    terminated_due_to_max_steps = terminated_due_to_max_steps or termination_reason == "budget_exhausted"
    final_score = float(env.state.final_score or 0.0)
    score = _clamp01(final_score)
    success = bool(observation is not None and observation.done and final_score > 0.0)
    result = {
        "task_id": task_id,
        "final_score": final_score,
        "score": score,
        "success": success,
        "total_reward": round(total_reward, 6),
        "steps": env.state.step_count,
        "derived_object_count": env.state.derived_object_count,
        "benchmark_runs": env.state.benchmark_runs,
        "final_message": (observation.message if observation is not None else (caught_error or "episode failed before first step")),
        "terminated_due_to_max_steps": terminated_due_to_max_steps,
        "done": bool(observation.done) if observation is not None else False,
        "benchmark_context": (observation.benchmark_context if observation is not None else {}),
        "router_summary": (getattr(observation, "router_summary", {}) or {}) if observation is not None else {},
        "decision_state": (getattr(observation, "decision_state", {}) or {}) if observation is not None else {},
        "final_feedback": final_feedback,
        "run_summary": {
            "model_name": model_name,
            "max_steps_config": max_steps,
            "effective_max_steps": effective_max_steps,
            "task_budget_max_steps": task_budget,
            "max_action_retries": max_action_retries,
            "termination_reason": termination_reason,
            "warnings": warnings,
            "trace_tail": action_trace[-10:],
        },
        "rewards": rewards,
    }

    _print_end_line(
        success=success,
        steps=env.state.step_count,
        score=score,
        rewards=rewards,
    )

    if output_path is None:
        safe_task_id = _sanitize_filename_component(task_id)
        safe_model_name = _sanitize_filename_component(model_name)
        result_path = Path(f"inference_result_{safe_task_id}_{safe_model_name}_v2.json")
    else:
        result_path = output_path
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run submission-style inference against schemaopt_env.")
    parser.add_argument("--tasks", default=None, help="Optional comma-separated task ids to run.")
    parser.add_argument("--task-id", default=None, help="Optional single task id override. Defaults to TASK_ID or the built-in default task when omitted.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model id for the OpenAI-compatible API.")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL, help="Optional OpenAI-compatible API base URL.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Optional external runner cap on steps. When omitted, the task budget is used as the episode limit.",
    )
    parser.add_argument("--max-action-retries", type=int, default=DEFAULT_MAX_ACTION_RETRIES, help="Maximum model retries per step.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for the result JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tasks:
        task_ids = _task_list_from_arg(args.tasks)
    elif args.task_id:
        task_ids = [args.task_id]
    else:
        task_ids = [DEFAULT_TASK_ID]
    results = [
        run_episode(
            task_id=task_id,
            model_name=args.model_name,
            api_base_url=args.api_base_url,
            max_steps=args.max_steps,
            max_action_retries=args.max_action_retries,
            output_path=(args.output if len(task_ids) == 1 else None),
        )
        for task_id in task_ids
    ]
    if args.output is not None and len(task_ids) > 1:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)


if __name__ == "__main__":
    main()
