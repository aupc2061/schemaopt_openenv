
"""Core schema optimization environment implementation backed by real DuckDB workloads."""

from __future__ import annotations

from collections import Counter
from numbers import Real
import copy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar
from uuid import uuid4

try:
    from models import SchemaOptAction, SchemaOptObservation, SchemaOptState
    from tasks import TASK_CATALOG, QuerySpec, TaskSpec, cluster_lookup, get_task, match_queries, query_lookup, resolve_runtime_asset_path, similar_query_ids, visible_queries_for_cluster
    from .rubrics import SchemaOptRubric
except ImportError:
    from models import SchemaOptAction, SchemaOptObservation, SchemaOptState
    from tasks import TASK_CATALOG, QuerySpec, TaskSpec, cluster_lookup, get_task, match_queries, query_lookup, resolve_runtime_asset_path, similar_query_ids, visible_queries_for_cluster
    from server.rubrics import SchemaOptRubric

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:
    try:
        from openenv_core.env_server.interfaces import Environment
        from openenv_core.env_server.types import State
    except ImportError:
        _A = TypeVar("_A")
        _O = TypeVar("_O")
        _S = TypeVar("_S")
        class Environment(Generic[_A, _O, _S]):
            def __init__(self, transform: Any = None, rubric: Any = None):
                self.transform = transform
                self.rubric = rubric

            def _apply_rubric(self, action: Any, observation: Any) -> float:
                return self.rubric(action, observation) if self.rubric is not None else 0.0

            def _reset_rubric(self) -> None:
                if self.rubric is not None:
                    self.rubric.reset()
        State = SchemaOptState

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BLOCKING = {"HASH_GROUP_BY", "PERFECT_HASH_GROUP_BY", "SORT", "WINDOW", "DISTINCT", "TOP_N", "ORDER_BY", "AGGREGATE"}

@dataclass
class PlanArtifact:
    raw_explain_text: str
    raw_explain_json: Any
    plan_depth: int
    operator_count: int
    join_count: int
    blocking_operator_count: int
    operators: List[str]
    def summary(self) -> Dict[str, Any]:
        return {"raw_explain_text": self.raw_explain_text, "plan_depth": self.plan_depth, "operator_count": self.operator_count, "join_count": self.join_count, "blocking_operator_count": self.blocking_operator_count, "operators": list(self.operators)}

@dataclass
class QueryExecution:
    runtime_ms: float
    plan: PlanArtifact
    column_names: List[str]
    rows: List[Tuple[Any, ...]]

@dataclass
class ParsedSQL:
    tables: List[str]
    canonical_tables: List[str]
    projection_aliases: List[str]
    group_by: List[str]
    raw_filter_predicates: List[str]
    canonical_filter_predicates: List[str]
    measure_columns: List[str]
    aggregate_functions: List[str]

@dataclass
class DerivedObject:
    name: str
    object_kind: str
    sql_definition: str
    source_objects: List[str]
    grain_dims: List[str]
    available_columns: List[str]
    canonical_columns: List[str]
    canonical_to_physical: Dict[str, str]
    column_types: Dict[str, str]
    parsed_sql: ParsedSQL
    row_count: int
    storage_bytes_estimate: int
    build_runtime_ms: float
    signature: str
    used_by_visible_queries: set[str] = field(default_factory=set)
    used_by_clusters: set[str] = field(default_factory=set)
    def summary(self) -> Dict[str, Any]:
        return {"name": self.name, "object_kind": self.object_kind, "source_objects": list(self.source_objects), "grain_dims": list(self.grain_dims), "available_columns": list(self.available_columns), "canonical_columns": list(self.canonical_columns), "row_count": self.row_count, "storage_bytes_estimate": self.storage_bytes_estimate, "build_runtime_ms": round(self.build_runtime_ms, 4), "signature": self.signature, "used_by_visible_queries": sorted(self.used_by_visible_queries), "used_by_clusters": sorted(self.used_by_clusters)}

class SchemaOptEnvironment(Environment[SchemaOptAction, SchemaOptObservation, SchemaOptState]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    LAST_GRADER_REPORT: Dict[str, Any] = {"available": False, "reason": "No episode executed yet."}
    GRADER_REPORTS_BY_TASK: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        try:
            super().__init__(rubric=SchemaOptRubric())
        except TypeError:
            try:
                super().__init__()
            except TypeError:
                pass
            self.rubric = SchemaOptRubric()
        first_task = next(iter(TASK_CATALOG.keys()), None)
        self._task: Optional[TaskSpec] = get_task(first_task) if first_task else None
        self._visible_query_lookup: Dict[str, QuerySpec] = {}
        self._all_query_lookup: Dict[str, QuerySpec] = {}
        self._cluster_lookup: Dict[str, Any] = {}
        self._derived_objects: Dict[str, DerivedObject] = {}
        self._retrieval_context: Dict[str, Any] = {}
        self._benchmark_context: Dict[str, Any] = {}
        self._router_summary: Dict[str, Any] = {}
        self._last_feedback: Dict[str, Any] = {}
        self._base_table_stats: Dict[str, Dict[str, Any]] = {}
        self._baseline_cache: Dict[str, QueryExecution] = {}
        self._evaluation_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._latest_visible_benchmark: Dict[str, float] = {"gated_improvement": 0.0, "correctness_coverage": 1.0}
        self._current_focus_cluster_id: Optional[str] = None
        self._state = SchemaOptState(episode_id=str(uuid4()), step_count=0)
        self._episode_root: Optional[Path] = None
        self._episode_db_path: Optional[Path] = None
        self._con: Any = None
        self._duckdb: Any = None

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_id: Optional[str] = None, **kwargs: Any) -> SchemaOptObservation:
        selected_task = task_id or kwargs.get("task_id") or next(iter(TASK_CATALOG.keys()))
        self._task = get_task(selected_task)
        self._visible_query_lookup = {q.query_id: q for q in self._task.visible_queries}
        self._all_query_lookup = query_lookup(self._task)
        self._cluster_lookup = cluster_lookup(self._task)
        self._derived_objects = {}
        self._retrieval_context = {"last_request": None, "matched_queries": [], "matched_clusters": [], "retrieval_count": 0}
        self._benchmark_context = {"baseline_weighted_cost": None, "current_weighted_cost": None, "raw_improvement": 0.0, "gated_improvement": 0.0, "correctness_coverage": 1.0, "routed_query_count": 0, "incorrect_query_count": 0, "last_benchmarked_query_ids": [], "last_benchmarked_cluster_id": None, "latest_plan_deltas": {}}
        self._router_summary = {"queries_routed": 0, "queries_unrouted": len(self._task.visible_queries), "dominant_rejection_reason": None, "candidate_object_coverage": {}, "last_scope": "reset"}
        self._last_feedback = {"event": "reset", "task_id": selected_task}
        self._baseline_cache = {}
        self._evaluation_cache = {}
        self._latest_visible_benchmark = {"gated_improvement": 0.0, "correctness_coverage": 1.0}
        self._current_focus_cluster_id = None
        self._reset_rubric()
        self._bootstrap_episode_database()
        self._base_table_stats = {table.name: self._collect_table_stats(table.name) for table in self._task.tables}
        self._state = SchemaOptState(episode_id=str(uuid4()) if episode_id is None else episode_id, step_count=0, done=False, task_id=self._task.task_id, difficulty=self._task.difficulty, derived_object_count=0, checkpoint_count=0, retrieval_count=0, benchmark_runs=0, storage_used_multiplier=0.0, final_score=None, last_error=None)
        self._refresh_public_state(None, "ok", {"event": "reset"}, False)
        payload = self._task.reset_payload()
        return SchemaOptObservation(status="ok", message=f"Initialized task {self._task.task_id}", decision_state=self._build_decision_state("ok", {"event": "reset"}, 0.0, False), catalog_summary=self._catalog_summary(payload), workload_summary=payload["workload_summary"], retrieval_context=self._retrieval_context, benchmark_context=self._benchmark_context, router_summary=self._router_summary, action_feedback=self._last_feedback, reward=0.0, done=False, metadata={"task": payload["task"]})

    @property
    def state(self) -> State:
        return self._state

    @classmethod
    def latest_report(cls, task_id: Optional[str] = None) -> Dict[str, Any]:
        if task_id:
            return cls.GRADER_REPORTS_BY_TASK.get(
                task_id,
                {"available": False, "task_id": task_id, "reason": f"No episode executed yet for task {task_id}."},
            )
        return cls.LAST_GRADER_REPORT

    @classmethod
    def run_baseline(cls) -> Dict[str, Any]:
        env = cls()
        env.reset(task_id=next(iter(TASK_CATALOG.keys())))
        cluster = env._task.clusters[0]
        query = visible_queries_for_cluster(env._task, cluster.cluster_id)[0]
        env.step(SchemaOptAction(operation="create_derived_object", object_kind=cluster.preferred_object_kind, name="baseline_object", sql_definition=query.sql, source_objects=list(query.tables), grain_hint=",".join(query.group_by)))
        env.step(SchemaOptAction(operation="benchmark_cluster", cluster_id=cluster.cluster_id))
        final_obs = env.step(SchemaOptAction(operation="submit"))
        return {"score": env.state.final_score, "reward": final_obs.reward, "done": final_obs.done, "message": final_obs.message, "benchmark": final_obs.benchmark_context, "rubric": final_obs.action_feedback}

    def _import_duckdb(self):
        if self._duckdb is not None:
            return self._duckdb
        try:
            import duckdb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("duckdb is required for schemaopt_env benchmark execution") from exc
        self._duckdb = duckdb
        return duckdb

    def _bootstrap_episode_database(self) -> None:
        duckdb = self._import_duckdb()
        if self._con is not None:
            self._con.close()
        if self._episode_root and self._episode_root.exists():
            shutil.rmtree(self._episode_root, ignore_errors=True)
        self._episode_root = Path(tempfile.mkdtemp(prefix="schemaopt_episodes_")) / str(uuid4())
        self._episode_root.mkdir(parents=True, exist_ok=True)
        source_db_path = Path(resolve_runtime_asset_path(self._task.database_path))
        if not source_db_path.exists():
            raise FileNotFoundError(f"Task database not found: {source_db_path}")
        self._episode_db_path = self._episode_root / source_db_path.name
        shutil.copy2(source_db_path, self._episode_db_path)
        self._con = duckdb.connect(str(self._episode_db_path))
        self._con.execute("CREATE SCHEMA IF NOT EXISTS derived")

    def _catalog_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        catalog = payload["catalog_summary"]
        catalog["tables"] = [self._base_table_stats.get(table.name, table.to_dict()) for table in self._task.tables]
        catalog["derived_objects"] = [obj.summary() for obj in self._derived_objects.values()]
        catalog["storage_usage_estimate"] = sum(obj.storage_bytes_estimate for obj in self._derived_objects.values())
        catalog["refresh_cost_estimate"] = round(sum(obj.build_runtime_ms for obj in self._derived_objects.values()), 4)
        return catalog

    def _build_observation(self, status: str, message: str, reward: float, done: bool, feedback: Dict[str, Any]) -> SchemaOptObservation:
        payload = self._task.reset_payload()
        return SchemaOptObservation(status=status, message=message, decision_state=self._build_decision_state(status, feedback, reward, done), catalog_summary=self._catalog_summary(payload), workload_summary=payload["workload_summary"], retrieval_context=self._retrieval_context, benchmark_context=self._benchmark_context, router_summary=self._router_summary, action_feedback=feedback, reward=round(reward, 6), done=done, metadata={"task": payload["task"]})

    def _step_budget_limit(self) -> int:
        return int(self._task.budgets.get("max_steps", 0) or 0)

    def _step_budget_exhausted(self) -> bool:
        step_budget = self._step_budget_limit()
        return step_budget > 0 and self._state.step_count >= step_budget

    def _finalize_due_to_budget(self, action: SchemaOptAction, status: str, message: str, feedback: Dict[str, Any]) -> Tuple[str, str, bool, Dict[str, Any], bool]:
        final_feedback = self._submit_episode()
        final_feedback["auto_submitted"] = True
        final_feedback["termination_reason"] = "budget_exhausted"
        final_feedback["originating_action"] = action.operation
        final_feedback["originating_status"] = status
        if status == "error":
            final_feedback["pre_terminal_error"] = feedback.get("error")
            final_feedback["pre_terminal_error_type"] = feedback.get("error_type")
        if message:
            final_message = f"{message} Step budget exhausted. Final benchmark submitted automatically."
        else:
            final_message = "Step budget exhausted. Final benchmark submitted automatically."
        return "completed", final_message, True, final_feedback, True

    def _reward_population_counts(self) -> Tuple[int, int]:
        return len(self._task.visible_queries), len(self._task.clusters)

    def _cluster_scope_key(self, cluster_id: str) -> str:
        return f"cluster:{cluster_id}"

    def _benchmark_score_snapshot(self, reward_inputs: Dict[str, Any]) -> float:
        return round(
            0.75 * float(reward_inputs.get("gated_improvement", 0.0))
            + 0.15 * float(reward_inputs.get("routed_query_ratio", 0.0))
            + 0.10 * float(reward_inputs.get("correctness_coverage", 1.0))
            - 0.50 * float(reward_inputs.get("budget_penalty", 0.0))
            - 0.10 * float(reward_inputs.get("resource_pressure", 0.0))
            - 0.05 * float(reward_inputs.get("incorrect_query_ratio", 0.0)),
            6,
        )

    def _cluster_for_query_ids(self, query_ids: Sequence[str]) -> Optional[str]:
        cluster_ids = {self._get_visible_query(query_id).cluster_id for query_id in query_ids}
        if len(cluster_ids) == 1:
            return next(iter(cluster_ids))
        return None

    def _focus_cluster_from_action(self, action: Optional[SchemaOptAction]) -> Optional[str]:
        if action is None:
            return self._current_focus_cluster_id
        if action.operation == "get_cluster_context" and action.cluster_id:
            return action.cluster_id
        if action.operation == "benchmark_cluster" and action.cluster_id:
            return action.cluster_id
        if action.operation == "benchmark_subset" and action.query_ids:
            return self._cluster_for_query_ids(action.query_ids)
        if action.operation in {"create_derived_object", "modify_derived_object"} and len(action.intended_clusters) == 1:
            return action.intended_clusters[0]
        if action.operation == "inspect_rewrite_status":
            if action.cluster_id:
                return action.cluster_id
            if action.target_id:
                query = self._visible_query_lookup.get(action.target_id)
                if query is not None:
                    return query.cluster_id
                return self._current_focus_cluster_id
            if action.query_ids:
                return self._cluster_for_query_ids(action.query_ids)
        return self._current_focus_cluster_id

    def _cluster_object_names(self, cluster_id: str) -> List[str]:
        names: List[str] = []
        for obj in self._derived_objects.values():
            if cluster_id in obj.used_by_clusters:
                names.append(obj.name)
                continue
            diagnostics = self._derived_object_diagnostics(obj, [cluster_id])
            if cluster_id in diagnostics.get("eligible_visible_clusters", []):
                names.append(obj.name)
        return sorted(set(names))

    def _refresh_public_state(self, action: Optional[SchemaOptAction], status: str, feedback: Dict[str, Any], done: bool) -> None:
        focus_cluster_id = self._focus_cluster_from_action(action)
        if focus_cluster_id is not None:
            self._current_focus_cluster_id = focus_cluster_id
        self._state.done = done
        self._state.derived_object_count = len(self._derived_objects)
        self._state.checkpoint_count = 0
        self._state.storage_used_multiplier = round(
            sum(obj.storage_bytes_estimate for obj in self._derived_objects.values())
            / max(1.0, float(self._task.budgets.get("max_storage_bytes", 1.0))),
            6,
        )
        self._state.last_action_operation = action.operation if action is not None else feedback.get("event")
        self._state.last_action_status = status
        self._state.current_focus_cluster_id = self._current_focus_cluster_id
        self._state.last_scope_key = feedback.get("scope_key") or self._router_summary.get("last_scope")
        self._state.last_scope_benchmark_score = (
            self._benchmark_score_snapshot(feedback.get("reward_inputs", {}))
            if feedback.get("scope_key") and feedback.get("reward_inputs")
            else None
        )
        if done:
            self._state.final_score = feedback.get("final_score")
        self._state.last_error = feedback.get("error") if status == "error" else None
        self._state.derived_object_names = sorted(self._derived_objects.keys())
        self._state.useful_derived_object_names = sorted(
            name for name, obj in self._derived_objects.items() if obj.used_by_visible_queries
        )
        self._state.unused_derived_object_names = sorted(
            name for name, obj in self._derived_objects.items() if not obj.used_by_visible_queries
        )
        max_steps = int(self._task.budgets.get("max_steps", 0))
        self._state.remaining_steps = max(0, max_steps - self._state.step_count)
        self._state.remaining_object_budget = max(
            0,
            int(self._task.budgets.get("max_new_derived_objects", 0)) - len(self._derived_objects),
        )
        self._state.remaining_storage_bytes = max(
            0,
            int(float(self._task.budgets.get("max_storage_bytes", 0)) - sum(obj.storage_bytes_estimate for obj in self._derived_objects.values())),
        )
        self._state.remaining_refresh_runtime_ms = max(
            0.0,
            round(
                float(self._task.budgets.get("max_refresh_runtime_ms", 0.0)) - sum(obj.build_runtime_ms for obj in self._derived_objects.values()),
                6,
            ),
        )
        self._state.resource_pressure = self._resource_pressure()

        previous_best = dict(getattr(self._state, "cluster_best_gated_improvement", {}))
        previous_routed = dict(getattr(self._state, "cluster_last_routed_query_count", {}))
        previous_incorrect = dict(getattr(self._state, "cluster_last_incorrect_query_count", {}))
        previous_score = dict(getattr(self._state, "cluster_last_benchmark_score", {}))
        previous_reason = dict(getattr(self._state, "cluster_dominant_rejection_reason", {}))
        previous_status = dict(getattr(self._state, "cluster_status_by_id", {}))
        previous_attempts = dict(getattr(self._state, "cluster_attempt_counts", {}))

        cluster_status_by_id: Dict[str, str] = {}
        cluster_attempt_counts: Dict[str, int] = {}
        cluster_best_gated_improvement: Dict[str, float] = {}
        cluster_last_routed_query_count: Dict[str, int] = {}
        cluster_last_incorrect_query_count: Dict[str, int] = {}
        cluster_last_benchmark_score: Dict[str, float] = {}
        cluster_dominant_rejection_reason: Dict[str, Optional[str]] = {}

        scoped_cluster_id: Optional[str] = None
        if feedback.get("scope_key", "").startswith("cluster:"):
            scoped_cluster_id = str(feedback["scope_key"]).split(":", 1)[1]
        elif self._current_focus_cluster_id is not None:
            scoped_cluster_id = self._current_focus_cluster_id

        for cluster in self._task.clusters:
            cluster_id = cluster.cluster_id
            object_names = self._cluster_object_names(cluster_id)
            prior_attempts = previous_attempts.get(cluster_id, 0)
            if action is not None and action.operation in {"create_derived_object", "modify_derived_object"} and cluster_id in action.intended_clusters:
                cluster_attempt_counts[cluster_id] = prior_attempts + 1
            else:
                cluster_attempt_counts[cluster_id] = prior_attempts

            cluster_best_gated_improvement[cluster_id] = previous_best.get(cluster_id, 0.0)
            cluster_last_routed_query_count[cluster_id] = previous_routed.get(cluster_id, 0)
            cluster_last_incorrect_query_count[cluster_id] = previous_incorrect.get(cluster_id, 0)
            cluster_last_benchmark_score[cluster_id] = previous_score.get(cluster_id, 0.0)
            cluster_dominant_rejection_reason[cluster_id] = previous_reason.get(cluster_id)
            status_label = previous_status.get(cluster_id, "untouched")

            if object_names and status_label == "untouched":
                status_label = "candidate_built"

            if status == "ok" and self._state.last_action_operation == "get_cluster_context" and self._current_focus_cluster_id == cluster_id:
                status_label = "analyzing"

            if feedback.get("scope_key") == self._cluster_scope_key(cluster_id):
                current_gated = float(feedback.get("gated_improvement") or self._benchmark_context.get("gated_improvement") or 0.0)
                current_routed = int(feedback.get("routed_query_count") or self._benchmark_context.get("routed_query_count") or 0)
                current_incorrect = int(feedback.get("incorrect_query_count") or self._benchmark_context.get("incorrect_query_count") or 0)
                current_score = self._benchmark_score_snapshot(feedback.get("reward_inputs", {})) if feedback.get("reward_inputs") else 0.0
                cluster_best_gated_improvement[cluster_id] = max(previous_best.get(cluster_id, 0.0), current_gated)
                cluster_last_routed_query_count[cluster_id] = current_routed
                cluster_last_incorrect_query_count[cluster_id] = current_incorrect
                cluster_last_benchmark_score[cluster_id] = current_score
                cluster_dominant_rejection_reason[cluster_id] = self._router_summary.get("dominant_rejection_reason")
                if current_gated > 0 and current_routed > 0 and current_incorrect == 0:
                    status_label = "verified_positive"
                else:
                    status_label = "verified_negative"
            elif (
                status == "ok"
                and scoped_cluster_id == cluster_id
                and self._state.last_action_operation in {"create_derived_object", "modify_derived_object", "drop_derived_object", "inspect_rewrite_status"}
            ):
                status_label = "candidate_built" if object_names else "analyzing"
                if self._router_summary.get("dominant_rejection_reason"):
                    cluster_dominant_rejection_reason[cluster_id] = self._router_summary.get("dominant_rejection_reason")
            elif self._current_focus_cluster_id == cluster_id and status_label == "untouched":
                status_label = "analyzing"

            if status == "error" and scoped_cluster_id == cluster_id:
                status_label = "blocked_error"
                cluster_dominant_rejection_reason[cluster_id] = feedback.get("error_type")

            cluster_status_by_id[cluster_id] = status_label

        self._state.cluster_status_by_id = cluster_status_by_id
        self._state.cluster_attempt_counts = cluster_attempt_counts
        self._state.cluster_best_gated_improvement = cluster_best_gated_improvement
        self._state.cluster_last_routed_query_count = cluster_last_routed_query_count
        self._state.cluster_last_incorrect_query_count = cluster_last_incorrect_query_count
        self._state.cluster_last_benchmark_score = cluster_last_benchmark_score
        self._state.cluster_dominant_rejection_reason = cluster_dominant_rejection_reason

    def _build_decision_state(self, status: str, feedback: Dict[str, Any], reward: float, done: bool) -> Dict[str, Any]:
        cluster_entries: List[Dict[str, Any]] = []
        verified_positive_clusters: List[str] = []
        unsolved_clusters: List[str] = []
        for cluster in self._task.clusters:
            cluster_id = cluster.cluster_id
            state_label = self._state.cluster_status_by_id.get(cluster_id, "untouched")
            if state_label == "verified_positive":
                verified_positive_clusters.append(cluster_id)
            elif state_label not in {"verified_positive", "verified_negative"}:
                unsolved_clusters.append(cluster_id)
            cluster_entries.append(
                {
                    "cluster_id": cluster_id,
                    "status": state_label,
                    "hotspot_rank": cluster.hotspot_rank,
                    "attempt_count": self._state.cluster_attempt_counts.get(cluster_id, 0),
                    "derived_object_names": self._cluster_object_names(cluster_id),
                    "best_gated_improvement": round(self._state.cluster_best_gated_improvement.get(cluster_id, 0.0), 6),
                    "last_routed_query_count": self._state.cluster_last_routed_query_count.get(cluster_id, 0),
                    "last_incorrect_query_count": self._state.cluster_last_incorrect_query_count.get(cluster_id, 0),
                    "last_benchmark_score": round(self._state.cluster_last_benchmark_score.get(cluster_id, 0.0), 6),
                    "dominant_rejection_reason": self._state.cluster_dominant_rejection_reason.get(cluster_id),
                }
            )

        focus_status = self._state.cluster_status_by_id.get(self._current_focus_cluster_id or "", "untouched")
        focus_has_objects = bool(self._current_focus_cluster_id and self._cluster_object_names(self._current_focus_cluster_id))
        all_resolved = all(
            self._state.cluster_status_by_id.get(cluster.cluster_id, "untouched") in {"verified_positive", "verified_negative"}
            for cluster in self._task.clusters
        )
        if done or (verified_positive_clusters and self._state.remaining_steps <= 3) or all_resolved:
            phase = "submit"
        elif self._state.last_action_operation in {"create_derived_object", "modify_derived_object", "drop_derived_object"}:
            phase = "validate"
        elif self._state.last_action_operation == "inspect_rewrite_status":
            phase = "validate"
        elif self._current_focus_cluster_id and not focus_has_objects:
            phase = "analyze"
        elif self._current_focus_cluster_id and focus_status in {"analyzing", "candidate_built", "blocked_error"}:
            phase = "build"
        else:
            phase = "analyze"

        affected_object_names: List[str] = []
        if "derived_object" in feedback:
            affected_object_names.append(feedback["derived_object"].get("name", ""))
        if "removed" in feedback:
            affected_object_names.append(feedback["removed"].get("name", ""))
        affected_object_names = [name for name in affected_object_names if name]

        last_rejection = None
        if status == "ok":
            last_rejection = self._router_summary.get("dominant_rejection_reason")
        else:
            last_rejection = feedback.get("error_type")

        return {
            "phase": phase,
            "current_focus_cluster_id": self._current_focus_cluster_id,
            "verified_positive_clusters": verified_positive_clusters,
            "unsolved_clusters": unsolved_clusters,
            "best_visible_gated_improvement": round(max(self._state.cluster_best_gated_improvement.values(), default=0.0), 6),
            "derived_objects": {
                "total_count": len(self._state.derived_object_names),
                "useful_count": len(self._state.useful_derived_object_names),
                "unused_count": len(self._state.unused_derived_object_names),
                "names": list(self._state.derived_object_names),
            },
            "clusters": cluster_entries,
            "remaining_budget_summary": {
                "steps_remaining": self._state.remaining_steps,
                "remaining_object_budget": self._state.remaining_object_budget,
                "remaining_storage_bytes": self._state.remaining_storage_bytes,
                "remaining_refresh_runtime_ms": self._state.remaining_refresh_runtime_ms,
                "resource_pressure": self._state.resource_pressure,
            },
            "last_action_effect": {
                "operation": self._state.last_action_operation,
                "status": status,
                "scoped_cluster_id": self._current_focus_cluster_id,
                "reward": reward,
                "routed_query_count": int(feedback.get("routed_query_count") or self._benchmark_context.get("routed_query_count") or 0),
                "gated_improvement": round(float(feedback.get("gated_improvement") or self._benchmark_context.get("gated_improvement") or 0.0), 6),
                "incorrect_query_count": int(feedback.get("incorrect_query_count") or self._benchmark_context.get("incorrect_query_count") or 0),
                "dominant_rejection_reason": last_rejection,
                "affected_object_names": affected_object_names,
            },
        }

    def _classify_error(self, exc: Exception) -> str:
        module_name = exc.__class__.__module__.lower()
        if "duckdb" in module_name:
            return "sql_runtime_error"
        return "internal_error"
    def _execute_action(self, action: SchemaOptAction, timeout_s: Optional[float] = None, **kwargs: Any) -> Tuple[str, str, bool, Dict[str, Any]]:
        done = False
        status = "ok"
        message = ""
        feedback: Dict[str, Any] = {}
        try:
            if action.operation == "inspect_catalog":
                feedback = {"event": "inspect_catalog", "base_tables": list(self._base_table_stats.values()), "derived_objects": [obj.summary() for obj in self._derived_objects.values()]}
                message = "Catalog summary returned."
            elif action.operation == "inspect_table_stats":
                feedback = {"event": "inspect_table_stats", "table": self._collect_table_stats(action.target_id or "")}
                message = f"Table stats returned for {action.target_id}."
            elif action.operation == "get_cluster_context":
                feedback = self._get_cluster_context(action.cluster_id or "", action.query_ids, action.top_k)
                self._state.retrieval_count += 1
                message = f"Cluster context returned for {action.cluster_id}."
            elif action.operation == "inspect_rewrite_status":
                feedback = self._inspect_rewrite_status(action)
                message = "Rewrite status returned."
            elif action.operation == "create_derived_object":
                feedback = self._upsert_derived_object(action, False)
                message = feedback.get("message", f"Created derived object {action.name}.")
            elif action.operation == "modify_derived_object":
                feedback = self._upsert_derived_object(action, True)
                message = feedback.get("message", f"Modified derived object {action.name}.")
            elif action.operation == "drop_derived_object":
                target_name = action.target_id or ""
                if target_name not in self._derived_objects:
                    raise ValueError(f"Derived object '{target_name}' does not exist")
                removed = self._derived_objects[target_name]
                pre_resource_pressure = self._resource_pressure()
                diagnostics = self._derived_object_diagnostics(removed, [])
                duplicate_like = any(obj.name != removed.name and obj.signature == removed.signature for obj in self._derived_objects.values())
                self._derived_objects.pop(target_name)
                self._con.execute(f"DROP TABLE IF EXISTS derived.{removed.name}")
                self._evaluation_cache = {}
                post_resource_pressure = self._resource_pressure()
                feedback = {
                    "event": "drop_derived_object",
                    "removed": removed.summary(),
                    "reward_inputs": {
                        "is_empty_object": diagnostics["is_empty_object"],
                        "duplicate_like": duplicate_like,
                        "eligible_visible_queries": diagnostics["eligible_visible_queries"],
                        "eligible_visible_cluster_count": len(diagnostics["eligible_visible_clusters"]),
                        "used_by_visible_queries_count": len(removed.used_by_visible_queries),
                        "used_by_cluster_count": len(removed.used_by_clusters),
                        "resource_pressure": post_resource_pressure,
                        "resource_pressure_delta": round(post_resource_pressure - pre_resource_pressure, 6),
                    },
                }
                message = f"Dropped derived object {action.target_id}."
            elif action.operation == "benchmark_subset":
                feedback = self._benchmark_action([self._get_visible_query(query_id) for query_id in action.query_ids], None)
                message = f"Benchmarked {len(action.query_ids)} visible queries."
            elif action.operation == "benchmark_cluster":
                feedback = self._benchmark_action(visible_queries_for_cluster(self._task, action.cluster_id or ""), action.cluster_id)
                message = f"Benchmarked cluster {action.cluster_id}."
            elif action.operation == "submit":
                feedback = self._submit_episode()
                status = "completed"
                done = True
                message = "Final benchmark submitted."
            else:
                raise ValueError(f"Unsupported action: {action.operation}")
        except ValueError as exc:
            status = "error"
            message = f"ERROR: {exc}"
            feedback = {"event": action.operation, "error": str(exc), "error_type": "validation_error"}
            self._state.last_error = str(exc)
        except Exception as exc:
            status = "error"
            message = f"ERROR: {exc}"
            feedback = {"event": action.operation, "error": str(exc), "error_type": self._classify_error(exc)}
            self._state.last_error = str(exc)
        return status, message, done, feedback

    def step(self, action: SchemaOptAction, timeout_s: Optional[float] = None, **kwargs: Any) -> SchemaOptObservation:
        self._state.step_count += 1
        status, message, done, feedback = self._execute_action(action, timeout_s=timeout_s, **kwargs)
        budget_terminated = False
        if not done and self._step_budget_exhausted():
            status, message, done, feedback, budget_terminated = self._finalize_due_to_budget(action, status, message, feedback)
        self._last_feedback = feedback
        self._refresh_public_state(action, status, feedback, done)
        observation = self._build_observation(status, message, 0.0, done, feedback)
        reward_action = SchemaOptAction(operation="submit") if budget_terminated else action
        observation.reward = round(self._apply_rubric(reward_action, observation), 6)
        if observation.decision_state.get("last_action_effect"):
            observation.decision_state["last_action_effect"]["reward"] = observation.reward
        return observation

    def _inspect_query_plan(self, query_id: str) -> Dict[str, Any]:
        query = self._get_visible_query(query_id)
        baseline = self._baseline_for_query(query)
        decision = self._evaluate_query(query, False)
        route_summary = self._route_summary(query_id, decision)
        self._router_summary = self._summarize_routes([route_summary], f"query:{query_id}")
        return {
            "event": "inspect_rewrite_status",
            "query_id": query_id,
            "baseline_plan": baseline.plan.summary(),
            "current_plan": route_summary,
            "top_rewrite_candidate": decision.get("top_rewrite_candidate"),
            "dominant_rejection_reason": route_summary.get("top_rejection_reason"),
        }

    def _inspect_router_status(self, query_ids: Sequence[str], cluster_id: Optional[str]) -> Dict[str, Any]:
        routes = [self._route_summary(query_id, self._evaluate_query(self._get_visible_query(query_id), False)) for query_id in query_ids]
        self._router_summary = self._summarize_routes(routes, f"cluster:{cluster_id}" if cluster_id else "explicit_query_set")
        return {"event": "inspect_rewrite_status", "routes": routes, "router_summary": self._router_summary}

    def _get_query_context(self, query_ids: Sequence[str]) -> Dict[str, Any]:
        contexts = [self._get_visible_query(query_id).context(similar_query_ids(self._task, query_id)) for query_id in query_ids]
        self._retrieval_context = {"last_request": {"mode": "get_cluster_context", "query_ids": list(query_ids)}, "matched_queries": [ctx["query_id"] for ctx in contexts], "matched_clusters": sorted({ctx["cluster_id"] for ctx in contexts}), "retrieval_count": self._state.retrieval_count + 1}
        return {"event": "get_cluster_context", "query_context": contexts}

    def _get_cluster_context(self, cluster_id: str, query_ids: Sequence[str], top_k: Optional[int]) -> Dict[str, Any]:
        if cluster_id not in self._cluster_lookup:
            raise ValueError(f"Unknown cluster_id: {cluster_id}")
        cluster = self._cluster_lookup[cluster_id]
        visible_cluster_queries = visible_queries_for_cluster(self._task, cluster_id)
        if query_ids:
            selected_queries = [self._get_visible_query(query_id) for query_id in query_ids]
            if any(query.cluster_id != cluster_id for query in selected_queries):
                raise ValueError("query_ids must belong to the requested cluster")
        else:
            selected_queries = list(visible_cluster_queries[: max(1, top_k or 2)])
        query_context = self._get_query_context([query.query_id for query in selected_queries])
        rewrite_status = self._inspect_router_status([query.query_id for query in selected_queries], cluster_id)
        relevant_objects = [
            obj.summary()
            for obj in self._derived_objects.values()
            if cluster_id in obj.used_by_clusters
            or cluster_id in self._derived_object_diagnostics(obj, [cluster_id]).get("eligible_visible_clusters", [])
        ]
        benchmark_snapshot = self._benchmark_context if self._benchmark_context.get("last_benchmarked_cluster_id") == cluster_id else {}
        self._current_focus_cluster_id = cluster_id
        self._retrieval_context = {
            "last_request": {"mode": "get_cluster_context", "cluster_id": cluster_id, "query_ids": [query.query_id for query in selected_queries], "top_k": top_k or 2},
            "matched_queries": [query.query_id for query in selected_queries],
            "matched_clusters": [cluster_id],
            "retrieval_count": self._state.retrieval_count + 1,
        }
        return {
            "event": "get_cluster_context",
            "cluster": cluster.to_summary(),
            "representative_queries": [query.summary() for query in selected_queries],
            "query_context": query_context["query_context"],
            "router_summary": rewrite_status["router_summary"],
            "routes": rewrite_status["routes"],
            "benchmark_snapshot": benchmark_snapshot,
            "relevant_derived_objects": relevant_objects,
        }

    def _inspect_rewrite_status(self, action: SchemaOptAction) -> Dict[str, Any]:
        if action.target_id:
            return self._inspect_query_plan(action.target_id)
        if action.cluster_id:
            query_ids = [query.query_id for query in visible_queries_for_cluster(self._task, action.cluster_id)]
            return self._inspect_router_status(query_ids, action.cluster_id)
        return self._inspect_router_status(list(action.query_ids), None)

    def _upsert_derived_object(self, action: SchemaOptAction, modify: bool) -> Dict[str, Any]:
        name = (action.name or "").strip()
        if not _IDENTIFIER_RE.match(name):
            raise ValueError(f"Invalid derived object name: {name}")
        if modify and name not in self._derived_objects:
            raise ValueError(f"Derived object '{name}' does not exist")
        if not modify and name in self._derived_objects:
            raise ValueError(f"Derived object '{name}' already exists")

        visible_query_count, visible_cluster_count = self._reward_population_counts()
        previous_obj = self._derived_objects.get(name) if modify else None
        previous_diagnostics = self._derived_object_diagnostics(previous_obj, action.intended_clusters) if previous_obj is not None else None
        pre_resource_pressure = self._resource_pressure()

        parsed = self._parse_sql_metadata(action.sql_definition or "")
        source_objects = [self._canonicalize_table_name(source) for source in action.source_objects]
        if sorted(parsed.canonical_tables) != sorted(source_objects):
            raise ValueError("source_objects must match sql_definition tables")
        grain_dims = [item.strip().lower() for item in (action.grain_hint or "").split(",") if item.strip()] or list(parsed.group_by)
        signature = self._signature_from_components(action.object_kind or "agg_matview", parsed, grain_dims)
        similar_existing = [obj.name for obj in self._derived_objects.values() if obj.name != name and obj.signature == signature]
        if similar_existing:
            self._router_summary = {
                "queries_routed": 0,
                "queries_unrouted": len(self._task.visible_queries),
                "dominant_rejection_reason": "duplicate_signature",
                "candidate_object_coverage": {existing: 0 for existing in similar_existing},
                "last_scope": f"duplicate:{name}",
            }
            return {
                "event": "modify_derived_object" if modify else "create_derived_object",
                "message": f"Skipped {name}: duplicate of existing derived object signature.",
                "duplicate_signature": True,
                "similar_existing_objects": similar_existing,
                "reward_inputs": {
                    "duplicate_signature": True,
                    "is_empty_object": False,
                    "eligible_visible_queries": 0,
                    "eligible_visible_cluster_count": 0,
                    "eligible_visible_clusters": [],
                    "visible_query_count": visible_query_count,
                    "visible_cluster_count": visible_cluster_count,
                    "resource_pressure": self._resource_pressure(),
                    "resource_pressure_delta": 0.0,
                    "previous_visible_query_count": visible_query_count,
                    "previous_visible_cluster_count": visible_cluster_count,
                    "previous_eligible_visible_queries": previous_diagnostics["eligible_visible_queries"] if previous_diagnostics else 0,
                    "previous_eligible_visible_cluster_count": len(previous_diagnostics["eligible_visible_clusters"]) if previous_diagnostics else 0,
                },
            }

        start_time = time.perf_counter()
        create_stmt = "CREATE OR REPLACE TABLE" if modify else "CREATE TABLE"
        self._con.execute(f"{create_stmt} derived.{name} AS ({action.sql_definition})")
        build_runtime_ms = (time.perf_counter() - start_time) * 1000.0
        describe_rows = self._con.execute(f"DESCRIBE derived.{name}").fetchall()
        column_types = {str(row[0]).lower(): str(row[1]) for row in describe_rows}
        physical_columns = list(column_types.keys())
        canonical_columns = [self._canonicalize_measure_name(column) for column in physical_columns]
        canonical_to_physical = {canonical: physical for physical, canonical in zip(physical_columns, canonical_columns)}
        row_count = int(self._con.execute(f"SELECT COUNT(*) FROM derived.{name}").fetchone()[0])
        obj = DerivedObject(
            name=name,
            object_kind=action.object_kind or "agg_matview",
            sql_definition=action.sql_definition or "",
            source_objects=source_objects,
            grain_dims=grain_dims,
            available_columns=physical_columns,
            canonical_columns=canonical_columns,
            canonical_to_physical=canonical_to_physical,
            column_types=column_types,
            parsed_sql=parsed,
            row_count=row_count,
            storage_bytes_estimate=self._estimate_storage_bytes(row_count, column_types),
            build_runtime_ms=build_runtime_ms,
            signature=signature,
        )
        self._derived_objects[name] = obj
        self._evaluation_cache = {}
        diagnostics = self._derived_object_diagnostics(obj, action.intended_clusters)
        post_resource_pressure = self._resource_pressure()
        self._router_summary = {
            "queries_routed": diagnostics["eligible_visible_queries"],
            "queries_unrouted": max(0, len(self._task.visible_queries) - diagnostics["eligible_visible_queries"]),
            "dominant_rejection_reason": diagnostics["top_rejection_reason"],
            "candidate_object_coverage": {obj.name: diagnostics["eligible_visible_queries"]},
            "last_scope": f"derived_object:{obj.name}",
        }
        return {
            "event": "modify_derived_object" if modify else "create_derived_object",
            "message": (f"Modified derived object {name}." if modify else f"Created derived object {name}."),
            "derived_object": obj.summary(),
            "reward_inputs": {
                "duplicate_signature": False,
                "is_empty_object": diagnostics["is_empty_object"],
                "eligible_visible_queries": diagnostics["eligible_visible_queries"],
                "eligible_visible_cluster_count": len(diagnostics["eligible_visible_clusters"]),
                "eligible_visible_clusters": list(diagnostics["eligible_visible_clusters"]),
                "visible_query_count": visible_query_count,
                "visible_cluster_count": visible_cluster_count,
                "resource_pressure": post_resource_pressure,
                "resource_pressure_delta": round(post_resource_pressure - pre_resource_pressure, 6),
                "previous_visible_query_count": visible_query_count,
                "previous_visible_cluster_count": visible_cluster_count,
                "previous_eligible_visible_queries": previous_diagnostics["eligible_visible_queries"] if previous_diagnostics else 0,
                "previous_eligible_visible_cluster_count": len(previous_diagnostics["eligible_visible_clusters"]) if previous_diagnostics else 0,
            },
            **diagnostics,
        }

    def _benchmark_action(self, queries: Sequence[QuerySpec], cluster_id: Optional[str]) -> Dict[str, Any]:
        summary = self._benchmark_queries(queries, True)
        self._latest_visible_benchmark = {"gated_improvement": summary["gated_improvement"], "correctness_coverage": summary["correctness_coverage"]}
        self._benchmark_context = {"baseline_weighted_cost": summary["baseline_weighted_cost"], "current_weighted_cost": summary["actual_current_weighted_cost"], "raw_improvement": summary["raw_improvement"], "gated_improvement": summary["gated_improvement"], "correctness_coverage": summary["correctness_coverage"], "routed_query_count": summary["routed_query_count"], "incorrect_query_count": summary["incorrect_query_count"], "last_benchmarked_query_ids": [query.query_id for query in queries], "last_benchmarked_cluster_id": cluster_id, "latest_plan_deltas": summary["plan_deltas"]}
        scope_key = f"cluster:{cluster_id}" if cluster_id else f"subset:{hashlib.md5('|'.join(sorted(query.query_id for query in queries)).encode('utf-8')).hexdigest()}"
        self._router_summary = self._summarize_routes(summary["per_query"], scope_key)
        self._state.benchmark_runs += 1
        scope_query_count = max(1, len(queries))
        routed_query_ratio = summary["routed_query_count"] / scope_query_count
        incorrect_query_ratio = summary["incorrect_query_count"] / scope_query_count
        summary["event"] = "benchmark_cluster" if cluster_id else "benchmark_subset"
        summary["scope_key"] = scope_key
        summary["scope_query_count"] = len(queries)
        summary["reward_inputs"] = {
            "derived_state_hash": self._derived_state_hash(),
            "gated_improvement": summary["gated_improvement"],
            "correctness_coverage": summary["correctness_coverage"],
            "budget_penalty": summary["budget_penalty"],
            "incorrect_query_count": summary["incorrect_query_count"],
            "incorrect_query_ratio": round(incorrect_query_ratio, 6),
            "routed_query_count": summary["routed_query_count"],
            "routed_query_ratio": round(routed_query_ratio, 6),
            "scope_query_count": len(queries),
            "resource_pressure": self._resource_pressure(),
            "resource_pressure_delta": 0.0,
        }
        summary["benchmark_score_inputs"] = {
            "gated_improvement": summary["gated_improvement"],
            "routed_query_ratio": round(routed_query_ratio, 6),
            "correctness_coverage": summary["correctness_coverage"],
            "budget_penalty": summary["budget_penalty"],
            "resource_pressure": self._resource_pressure(),
            "incorrect_query_ratio": round(incorrect_query_ratio, 6),
        }
        return summary

    def _submit_episode(self) -> Dict[str, Any]:
        self._reset_visible_usage()
        visible = self._benchmark_queries(self._task.visible_queries, True)
        holdout = self._benchmark_queries(self._task.holdout_queries, False)
        migration = self._migration_score()
        storage = self._storage_efficiency_score()
        correctness = round((visible["correctness_coverage"] * 0.6) + (holdout["correctness_coverage"] * 0.4), 6)
        final_score = round(0.45 * visible["gated_improvement"] + 0.20 * holdout["gated_improvement"] + 0.20 * correctness + 0.10 * migration + 0.05 * storage, 6)
        self._benchmark_context = {"baseline_weighted_cost": visible["baseline_weighted_cost"], "current_weighted_cost": visible["actual_current_weighted_cost"], "raw_improvement": visible["raw_improvement"], "gated_improvement": visible["gated_improvement"], "correctness_coverage": visible["correctness_coverage"], "routed_query_count": visible["routed_query_count"], "incorrect_query_count": visible["incorrect_query_count"], "last_benchmarked_query_ids": [query.query_id for query in self._task.visible_queries], "last_benchmarked_cluster_id": None, "latest_plan_deltas": visible["plan_deltas"]}
        self._router_summary = self._summarize_routes(visible["per_query"], "submit")
        self._state.benchmark_runs += 1
        self._state.final_score = final_score
        grader_report = {
            "available": True,
            "task_id": self._task.task_id,
            "episode_id": self._state.episode_id,
            "score": final_score,
            "resolved": final_score > 0.0,
            "details": {
                "visible_summary": visible,
                "holdout_summary": holdout,
                "migration_score": migration,
                "storage_score": storage,
            },
        }
        SchemaOptEnvironment.LAST_GRADER_REPORT = grader_report
        SchemaOptEnvironment.GRADER_REPORTS_BY_TASK[self._task.task_id] = grader_report
        return {"event": "submit", "final_score": final_score, "visible_summary": visible, "holdout_summary": holdout, "migration_score": migration, "storage_score": storage, "reward_inputs": {"final_score": final_score, "visible_gated_improvement": visible["gated_improvement"], "holdout_gated_improvement": holdout["gated_improvement"], "correctness": correctness, "migration": migration, "storage": storage}, "final_score_inputs": {"visible_gated_improvement": visible["gated_improvement"], "holdout_gated_improvement": holdout["gated_improvement"], "correctness": correctness, "migration": migration, "storage": storage}}

    def _benchmark_queries(self, queries: Sequence[QuerySpec], mark_usage: bool) -> Dict[str, Any]:
        baseline_weighted_cost = actual_current_weighted_cost = gated_current_weighted_cost = weight_total = correctness_weight_total = 0.0
        routed_query_count = incorrect_query_count = 0
        deltas = {"depth_delta": 0.0, "operator_delta": 0.0, "runtime_delta_ms": 0.0}
        per_query: List[Dict[str, Any]] = []
        rejection_histogram: Counter[str] = Counter()
        routed_object_counter: Counter[str] = Counter()
        for query in queries:
            baseline = self._baseline_for_query(query)
            current = self._evaluate_query(query, mark_usage)
            baseline_cost = self._compose_query_cost(baseline.runtime_ms, baseline.plan)
            current_cost = self._compose_query_cost(current["runtime_ms"], current["plan"])
            weight = query.frequency_weight * query.priority_weight
            baseline_weighted_cost += baseline_cost * weight
            actual_current_weighted_cost += current_cost * weight
            gated_current_weighted_cost += (current_cost if current["correctness_pass"] else baseline_cost) * weight
            weight_total += weight
            correctness_weight_total += weight if current["correctness_pass"] else 0.0
            routed_query_count += 1 if current["routed"] else 0
            incorrect_query_count += 0 if current["correctness_pass"] else 1
            deltas["depth_delta"] += baseline.plan.plan_depth - current["plan"].plan_depth
            deltas["operator_delta"] += baseline.plan.operator_count - current["plan"].operator_count
            deltas["runtime_delta_ms"] += baseline.runtime_ms - current["runtime_ms"]
            route_summary = self._route_summary(query.query_id, current)
            per_query.append(route_summary)
            if route_summary["routed"] and route_summary["object_name"]:
                routed_object_counter[route_summary["object_name"]] += 1
            for item in route_summary.get("rejection_reasons", []):
                rejection_histogram[item.get("reason") or "rewrite_not_applicable"] += 1
        raw_improvement = 0.0 if baseline_weighted_cost == 0 else max(0.0, 1.0 - (actual_current_weighted_cost / baseline_weighted_cost))
        gated_improvement = 0.0 if baseline_weighted_cost == 0 else max(0.0, 1.0 - (gated_current_weighted_cost / baseline_weighted_cost))
        correctness_coverage = 1.0 if weight_total == 0 else correctness_weight_total / weight_total
        best_candidate_object = routed_object_counter.most_common(1)[0][0] if routed_object_counter else None
        return {"baseline_weighted_cost": round(baseline_weighted_cost, 6), "actual_current_weighted_cost": round(actual_current_weighted_cost, 6), "gated_current_weighted_cost": round(gated_current_weighted_cost, 6), "raw_improvement": round(raw_improvement, 6), "gated_improvement": round(gated_improvement, 6), "correctness_coverage": round(correctness_coverage, 6), "routed_query_count": routed_query_count, "incorrect_query_count": incorrect_query_count, "budget_penalty": round(self._budget_penalty(), 6), "plan_deltas": {key: round(value, 4) for key, value in deltas.items()}, "per_query": per_query, "unused_derived_objects": sorted(name for name, obj in self._derived_objects.items() if not obj.used_by_visible_queries), "best_candidate_object": best_candidate_object, "rejection_reason_histogram": dict(sorted(rejection_histogram.items()))}

    def _baseline_for_query(self, query: QuerySpec) -> QueryExecution:
        if query.query_id not in self._baseline_cache:
            self._baseline_cache[query.query_id] = self._execute_query(query.sql)
        return self._baseline_cache[query.query_id]

    def _evaluate_query(self, query: QuerySpec, mark_usage: bool) -> Dict[str, Any]:
        state_key = (query.query_id, self._derived_state_hash())
        if state_key in self._evaluation_cache:
            cached = copy.deepcopy(self._evaluation_cache[state_key])
            if mark_usage and cached["routed"] and cached["object_name"]:
                self._mark_usage(cached["object_name"], query)
            return cached
        baseline = self._baseline_for_query(query)
        best = {
            "routed": False,
            "object_name": None,
            "rewritten_sql": None,
            "runtime_ms": baseline.runtime_ms,
            "correctness_pass": True,
            "plan": baseline.plan,
            "route_reason": "base plan",
            "rejection_reasons": [],
            "top_rewrite_candidate": None,
        }
        best_correct_runtime = baseline.runtime_ms
        best_incorrect_runtime = baseline.runtime_ms
        for obj in self._derived_objects.values():
            rewrite, rejection_reason = self._build_rewrite(query, obj)
            if rewrite is None:
                best["rejection_reasons"].append({"object_name": obj.name, "reason": rejection_reason or "rewrite_not_applicable"})
                continue
            current = self._execute_query(rewrite["sql"])
            correctness = self._compare_results(baseline, current)
            candidate = {
                "routed": True,
                "object_name": obj.name,
                "rewritten_sql": rewrite["sql"],
                "runtime_ms": current.runtime_ms,
                "correctness_pass": correctness,
                "plan": current.plan,
                "route_reason": rewrite["reason"],
                "rejection_reasons": list(best["rejection_reasons"]),
                "top_rewrite_candidate": {"object_name": obj.name, "reason": rewrite["reason"], "runtime_ms": round(current.runtime_ms, 6), "correctness_pass": correctness},
            }
            if correctness and current.runtime_ms < best_correct_runtime:
                best = candidate
                best_correct_runtime = current.runtime_ms
            elif best["object_name"] is None and current.runtime_ms < best_incorrect_runtime:
                best = candidate
                best_incorrect_runtime = current.runtime_ms
        if not best["routed"] and best["rejection_reasons"]:
            best["top_rewrite_candidate"] = best["rejection_reasons"][0]
        self._evaluation_cache[state_key] = copy.deepcopy(best)
        if mark_usage and best["routed"] and best["object_name"]:
            self._mark_usage(best["object_name"], query)
        return best

    def _execute_query(self, sql: str, repeats: int = 3, warmup: int = 1) -> QueryExecution:
        timings: List[float] = []
        rows: List[Tuple[Any, ...]] = []
        columns: List[str] = []
        for idx in range(warmup + repeats):
            start = time.perf_counter()
            cursor = self._con.execute(sql)
            fetched = cursor.fetchall()
            elapsed = (time.perf_counter() - start) * 1000.0
            if idx >= warmup:
                timings.append(elapsed)
                rows = [tuple(row) for row in fetched]
                columns = [str(item[0]).lower() for item in cursor.description or []]
        return QueryExecution(runtime_ms=sum(timings) / max(1, len(timings)), plan=self._collect_plan(sql), column_names=columns, rows=rows)

    def _collect_plan(self, sql: str) -> PlanArtifact:
        raw_json = None
        raw_text = ""
        try:
            rows = self._con.execute(f"EXPLAIN (FORMAT json) {sql}").fetchall()
            raw_text = self._flatten_explain(rows)
            raw_json = json.loads(raw_text)
        except Exception:
            rows = self._con.execute(f"EXPLAIN {sql}").fetchall()
            raw_text = self._flatten_explain(rows)
        depth, operators, joins, blocking, names = self._summarize_plan(raw_json, raw_text)
        return PlanArtifact(raw_explain_text=raw_text, raw_explain_json=raw_json, plan_depth=depth, operator_count=operators, join_count=joins, blocking_operator_count=blocking, operators=names)

    def _build_rewrite(self, query: QuerySpec, obj: DerivedObject) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        if obj.row_count == 0:
            return None, "empty_derived_object"
        if sorted(query.canonical_tables) != sorted(obj.parsed_sql.canonical_tables):
            return None, "table_mismatch"
        if set(query.canonical_filter_predicates) != set(obj.parsed_sql.canonical_filter_predicates):
            return None, "predicate_mismatch"
        query_dims = [item.lower() for item in query.group_by]
        if not set(query_dims).issubset(set(obj.parsed_sql.group_by)):
            return None, "group_by_not_subset"
        if not set(query.measure_columns).issubset(set(obj.canonical_columns)):
            return None, "measure_columns_missing"
        exact_route, exact_rejection = self._build_exact_rewrite(query, obj)
        if exact_route is not None:
            return exact_route, None
        if obj.object_kind not in {"agg_matview", "denorm_table", "join_matview"}:
            return None, "object_kind_not_rollup_compatible"
        if query_dims == obj.parsed_sql.group_by:
            return None, exact_rejection or "rollup_not_applicable_exact_grain"
        if not all(func in {"count", "sum"} for func in query.aggregate_functions):
            return None, "aggregate_functions_not_supported"
        if query.order_by or query.limit is not None:
            return None, exact_rejection or "order_by_not_reconstructible"
        select_cols = list(query_dims)
        for canonical_name, result_label in zip(query.measure_columns, query.result_columns[len(query_dims):]):
            physical = obj.canonical_to_physical.get(canonical_name)
            if not physical:
                return None, "measure_columns_missing"
            select_cols.append(f"SUM({self._quote_identifier(physical)}) AS {self._quote_identifier(result_label)}")
        sql = f"SELECT {', '.join(select_cols)} FROM derived.{obj.name}"
        if query_dims:
            sql += " GROUP BY " + ", ".join(query_dims)
        return {"sql": sql, "reason": "rollup over derived object"}, None

    def _route_summary(self, query_id: str, route: Dict[str, Any]) -> Dict[str, Any]:
        rejection_reasons = list(route.get("rejection_reasons", []))
        top_rejection_reason = rejection_reasons[0]["reason"] if rejection_reasons else None
        return {
            "query_id": query_id,
            "routed": route["routed"],
            "object_name": route["object_name"],
            "rewritten_sql": route["rewritten_sql"],
            "runtime_ms": round(route["runtime_ms"], 6),
            "correctness_pass": route["correctness_pass"],
            "route_reason": route["route_reason"],
            "rejection_reasons": rejection_reasons,
            "top_rewrite_candidate": route.get("top_rewrite_candidate"),
            "top_rejection_reason": top_rejection_reason,
            "plan_depth": route["plan"].plan_depth,
            "operator_count": route["plan"].operator_count,
            "join_count": route["plan"].join_count,
            "blocking_operator_count": route["plan"].blocking_operator_count,
            "operators": list(route["plan"].operators),
        }
    def _compare_results(self, baseline: QueryExecution, current: QueryExecution) -> bool:
        if baseline.column_names != current.column_names or len(baseline.rows) != len(current.rows):
            return False
        left = sorted(self._normalize_row(row) for row in baseline.rows)
        right = sorted(self._normalize_row(row) for row in current.rows)
        return left == right

    def _normalize_row(self, row: Tuple[Any, ...]) -> Tuple[Any, ...]:
        result: List[Any] = []
        for value in row:
            if isinstance(value, Real) and not isinstance(value, bool):
                result.append(("number", round(float(value), 6)))
            else:
                result.append(value)
        return tuple(result)

    def _compose_query_cost(self, runtime_ms: float, plan: PlanArtifact) -> float:
        return 0.70 * runtime_ms + 5.0 * plan.plan_depth + 2.0 * plan.operator_count + 3.0 * plan.join_count + 2.5 * plan.blocking_operator_count

    def _migration_score(self) -> float:
        max_objects = max(1, int(self._task.budgets["max_new_derived_objects"]))
        object_ratio = len(self._derived_objects) / max_objects
        unused_ratio = 0.0 if not self._derived_objects else len([obj for obj in self._derived_objects.values() if not obj.used_by_visible_queries]) / len(self._derived_objects)
        return round(max(0.0, 1.0 - (0.35 * object_ratio) - (0.65 * unused_ratio)), 6)

    def _storage_efficiency_score(self) -> float:
        storage_limit = max(float(self._task.budgets.get("max_storage_bytes", 1.0)), 1.0)
        refresh_limit = max(float(self._task.budgets.get("max_refresh_runtime_ms", 1.0)), 1.0)
        storage_ratio = sum(obj.storage_bytes_estimate for obj in self._derived_objects.values()) / storage_limit
        refresh_ratio = sum(obj.build_runtime_ms for obj in self._derived_objects.values()) / refresh_limit
        return round(max(0.0, 1.0 - (0.50 * storage_ratio) - (0.50 * refresh_ratio)), 6)

    def _resource_pressure(self) -> float:
        storage_limit = max(float(self._task.budgets.get("max_storage_bytes", 1.0)), 1.0)
        refresh_limit = max(float(self._task.budgets.get("max_refresh_runtime_ms", 1.0)), 1.0)
        object_limit = max(int(self._task.budgets.get("max_new_derived_objects", 1)), 1)
        storage_ratio = sum(obj.storage_bytes_estimate for obj in self._derived_objects.values()) / storage_limit
        refresh_ratio = sum(obj.build_runtime_ms for obj in self._derived_objects.values()) / refresh_limit
        object_ratio = len(self._derived_objects) / object_limit
        return round(min(1.0, 0.4 * storage_ratio + 0.4 * refresh_ratio + 0.2 * object_ratio), 6)

    def _budget_penalty(self) -> float:
        storage_limit = max(float(self._task.budgets.get("max_storage_bytes", 1.0)), 1.0)
        refresh_limit = max(float(self._task.budgets.get("max_refresh_runtime_ms", 1.0)), 1.0)
        used_storage = sum(obj.storage_bytes_estimate for obj in self._derived_objects.values())
        used_refresh = sum(obj.build_runtime_ms for obj in self._derived_objects.values())
        penalty = 0.0
        if used_storage > storage_limit:
            penalty += min(0.15, (used_storage - storage_limit) / storage_limit)
        if used_refresh > refresh_limit:
            penalty += min(0.15, (used_refresh - refresh_limit) / refresh_limit)
        if len(self._derived_objects) > int(self._task.budgets["max_new_derived_objects"]):
            penalty += 0.10
        return penalty

    def _resolve_retrieval_mode(self, action: SchemaOptAction) -> str:
        if action.pattern and action.top_k is None and not any([action.cluster_id, action.tables, action.columns, action.plan_features]):
            return "regex"
        if action.pattern and action.top_k is not None:
            return "substring"
        if action.cluster_id:
            return "cluster"
        if action.tables:
            return "table_filter"
        if action.columns:
            return "column_filter"
        if action.plan_features:
            return "plan_filter"
        return "hotspot_rank"

    def _get_visible_query(self, query_id: str) -> QuerySpec:
        if query_id not in self._visible_query_lookup:
            raise ValueError(f"Unknown visible query_id: {query_id}")
        return self._visible_query_lookup[query_id]

    def _reset_visible_usage(self) -> None:
        for obj in self._derived_objects.values():
            obj.used_by_visible_queries.clear()
            obj.used_by_clusters.clear()

    def _mark_usage(self, object_name: str, query: QuerySpec) -> None:
        obj = self._derived_objects[object_name]
        obj.used_by_visible_queries.add(query.query_id)
        obj.used_by_clusters.add(query.cluster_id)

    def _derived_state_hash(self) -> str:
        material = "|".join(f"{name}:{hashlib.md5(obj.sql_definition.encode('utf-8')).hexdigest()}" for name, obj in sorted(self._derived_objects.items()))
        return hashlib.md5(material.encode("utf-8")).hexdigest()

    def _collect_table_stats(self, table_name: str) -> Dict[str, Any]:
        row_count = int(self._con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        describe_rows = self._con.execute(f"DESCRIBE {table_name}").fetchall()
        columns = [{"name": str(row[0]), "type": str(row[1])} for row in describe_rows]
        null_counts = {}
        for row in describe_rows[: min(5, len(describe_rows))]:
            name = str(row[0])
            null_counts[name] = int(self._con.execute(f'SELECT COUNT(*) FROM {table_name} WHERE "{name}" IS NULL').fetchone()[0])
        return {"name": table_name, "columns": columns, "row_count": row_count, "null_counts": null_counts}

    def _flatten_explain(self, rows: Sequence[Sequence[Any]]) -> str:
        parts: List[str] = []
        for row in rows:
            parts.append(str(row[-1]))
        return "\n".join(parts)

    def _summarize_plan(self, raw_json: Any, raw_text: str) -> Tuple[int, int, int, int, List[str]]:
        if raw_json is not None:
            depth, ops, joins, blocking, names = self._walk_plan(raw_json)
            if ops > 0:
                return depth, ops, joins, blocking, names
        names = []
        for line in raw_text.splitlines():
            token = line.strip(" │┌┐└┘─").upper().replace(" ", "_")
            if token and any(ch.isalpha() for ch in token) and token not in {"PHYSICAL_PLAN", "EXPLAIN_ANALYZE"}:
                names.append(token)
        joins = sum(1 for item in names if "JOIN" in item)
        blocking = sum(1 for item in names if item in _BLOCKING or "AGGREGATE" in item)
        return max(1, len(names)), len(names), joins, blocking, names

    def _walk_plan(self, node: Any) -> Tuple[int, int, int, int, List[str]]:
        if isinstance(node, list):
            summaries = [self._walk_plan(item) for item in node]
            if not summaries:
                return 1, 0, 0, 0, []
            return max(item[0] for item in summaries), sum(item[1] for item in summaries), sum(item[2] for item in summaries), sum(item[3] for item in summaries), [name for item in summaries for name in item[4]]
        if isinstance(node, dict):
            name = str(node.get("name") or node.get("operator_name") or node.get("operator_type") or node.get("type") or "").upper()
            children = node.get("children") or node.get("child") or node.get("plans") or []
            child = self._walk_plan(children) if children else (0, 0, 0, 0, [])
            return max(1, child[0] + 1 if children else 1), child[1] + (1 if name else 0), child[2] + (1 if "JOIN" in name else 0), child[3] + (1 if name in _BLOCKING or "AGGREGATE" in name else 0), ([name] if name else []) + child[4]
        return 1, 0, 0, 0, []

    def _estimate_storage_bytes(self, row_count: int, column_types: Dict[str, str]) -> int:
        sizes = {"BOOLEAN": 1, "TINYINT": 1, "SMALLINT": 2, "INTEGER": 4, "BIGINT": 8, "FLOAT": 4, "REAL": 4, "DOUBLE": 8, "DECIMAL": 16, "TIMESTAMP": 8, "DATE": 4, "VARCHAR": 24, "TEXT": 24}
        per_row = 0
        for dtype in column_types.values():
            upper = dtype.upper()
            per_row += next((value for key, value in sizes.items() if key in upper), 16)
        return int(max(1, row_count) * max(1, per_row))

    def _canonicalize_table_name(self, table_name: str) -> str:
        return table_name.replace('"', '').strip().lower()

    def _signature_from_components(self, object_kind: str, parsed: ParsedSQL, grain_dims: Sequence[str]) -> str:
        material = {
            "object_kind": object_kind,
            "tables": sorted(parsed.canonical_tables),
            "predicates": sorted(parsed.canonical_filter_predicates),
            "group_by": [item.lower() for item in grain_dims],
            "measures": sorted(parsed.measure_columns),
            "aggregate_functions": sorted(parsed.aggregate_functions),
        }
        return hashlib.md5(json.dumps(material, sort_keys=True).encode('utf-8')).hexdigest()

    def _derived_object_diagnostics(self, obj: DerivedObject, intended_clusters: Sequence[str]) -> Dict[str, Any]:
        target_queries = [query for query in self._task.visible_queries if not intended_clusters or query.cluster_id in intended_clusters]
        matched_query_ids: List[str] = []
        rejection_histogram: Counter[str] = Counter()
        matched_clusters: set[str] = set()
        for query in target_queries:
            rewrite, rejection_reason = self._build_rewrite(query, obj)
            if rewrite is not None:
                matched_query_ids.append(query.query_id)
                matched_clusters.add(query.cluster_id)
            else:
                rejection_histogram[rejection_reason or 'rewrite_not_applicable'] += 1
        similar_existing = [existing.name for existing in self._derived_objects.values() if existing.name != obj.name and existing.signature == obj.signature]
        top_rejection_reason = rejection_histogram.most_common(1)[0][0] if rejection_histogram else None
        return {
            "eligible_visible_queries": len(matched_query_ids),
            "eligible_visible_clusters": sorted(matched_clusters),
            "matched_query_ids": matched_query_ids,
            "top_rejection_reasons": dict(rejection_histogram.most_common(5)),
            "top_rejection_reason": top_rejection_reason,
            "is_empty_object": obj.row_count == 0,
            "similar_existing_objects": similar_existing,
        }

    def _summarize_routes(self, routes: Sequence[Dict[str, Any]], scope: str) -> Dict[str, Any]:
        rejection_histogram: Counter[str] = Counter()
        object_coverage: Counter[str] = Counter()
        routed_count = 0
        for route in routes:
            if route.get("routed"):
                routed_count += 1
            if route.get("object_name"):
                object_coverage[route["object_name"]] += 1
            for rejection in route.get("rejection_reasons", []):
                rejection_histogram[rejection.get("reason") or "rewrite_not_applicable"] += 1
        return {
            "queries_routed": routed_count,
            "queries_unrouted": max(0, len(routes) - routed_count),
            "dominant_rejection_reason": rejection_histogram.most_common(1)[0][0] if rejection_histogram else None,
            "candidate_object_coverage": dict(sorted(object_coverage.items())),
            "last_scope": scope,
        }

    def _parse_sql_metadata(self, sql: str) -> ParsedSQL:
        normalized = re.sub(r'\s+', ' ', sql.strip().rstrip(';')).strip()
        lowered = normalized.lower()
        select_match = re.search(r'\bselect\b', lowered)
        from_match = re.search(r'\bfrom\b', lowered)
        if not select_match or not from_match or from_match.start() <= select_match.end():
            raise ValueError("Only SELECT-derived objects are supported")
        select_clause = normalized[select_match.end():from_match.start()].strip()
        after_from = normalized[from_match.end():].strip()
        where_match = re.search(r"\bwhere\b", after_from, re.IGNORECASE)
        group_match = re.search(r"\bgroup\s+by\b", after_from, re.IGNORECASE)
        order_match = re.search(r"\border\s+by\b", after_from, re.IGNORECASE)
        limit_match = re.search(r"\blimit\b", after_from, re.IGNORECASE)
        where_end = len(after_from)
        for match in [group_match, order_match, limit_match]:
            if match and match.start() < where_end:
                where_end = match.start()
        where_clause = after_from[where_match.end():where_end].strip() if where_match else ""
        group_end = len(after_from)
        for match in [order_match, limit_match]:
            if match and match.start() < group_end:
                group_end = match.start()
        group_clause = after_from[group_match.end():group_end].strip() if group_match else ""
        parts = self._split_sql_list(select_clause)
        aliases = [((self._extract_alias(part) or self._default_result_label(part)).lower()) for part in parts]
        measures = [self._canonicalize_measure_name(self._extract_alias(part) or self._default_result_label(part)) for part in parts if self._aggregate_function(part)]
        funcs = [self._aggregate_function(part) for part in parts if self._aggregate_function(part)]
        raw_filter_predicates = self._split_predicates(where_clause)
        tables = [item.strip('"').lower() for item in re.findall(r'(?:from|join)\s+([A-Za-z0-9_\."]+)', normalized, flags=re.IGNORECASE)]
        return ParsedSQL(
            tables=tables,
            canonical_tables=[self._canonicalize_table_name(item) for item in tables],
            projection_aliases=aliases,
            group_by=[item.lower() for item in self._parse_group_by(group_clause, aliases)],
            raw_filter_predicates=raw_filter_predicates,
            canonical_filter_predicates=[self._normalize_predicate(item) for item in raw_filter_predicates],
            measure_columns=measures,
            aggregate_functions=[item.lower() for item in funcs if item],
        )

    def _split_sql_list(self, clause: str) -> List[str]:
        parts: List[str] = []
        current: List[str] = []
        depth = 0
        for char in clause:
            if char == '(':
                depth += 1
            elif char == ')':
                depth = max(0, depth - 1)
            if char == ',' and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current).strip())
        return parts

    def _extract_alias(self, expression: str) -> Optional[str]:
        match = re.search(r'\bas\s+(?:"([^"]+)"|([A-Za-z_][A-Za-z0-9_]*))\s*$', expression, re.IGNORECASE)
        if not match:
            return None
        return (match.group(1) or match.group(2) or '').lower()

    def _aggregate_function(self, expression: str) -> Optional[str]:
        match = re.search(r'\b(count|sum|avg|min|max)\s*\(', expression, re.IGNORECASE)
        return match.group(1).lower() if match else None

    def _split_predicates(self, clause: str) -> List[str]:
        clause = clause.strip()
        return [part.strip() for part in re.split(r'\s+AND\s+', clause, flags=re.IGNORECASE) if part.strip()] if clause else []

    def _normalize_predicate(self, predicate: str) -> str:
        normalized = " ".join(predicate.strip().lower().split())
        normalized = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.((?:"[^"]+")|(?:[a-zA-Z_][a-zA-Z0-9_]*))', lambda match: match.group(1), normalized)
        normalized = normalized.replace("\"", "")
        normalized = normalized.replace("( ", "(").replace(" )", ")")
        return normalized

    def _canonicalize_measure_name(self, value: str) -> str:
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


    def _default_result_label(self, value: str) -> str:
        normalized = value.strip().lower().replace('"', '')
        normalized = re.sub(r'\s+', ' ', normalized)
        if normalized == 'count(*)':
            return 'count_star()'
        return normalized

    def _quote_identifier(self, value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    def _build_exact_rewrite(self, query: QuerySpec, obj: DerivedObject) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        if tuple(query.group_by) != tuple(obj.parsed_sql.group_by):
            return None, 'exact_group_by_mismatch'
        if tuple(query.canonical_output_columns) != tuple(obj.canonical_columns):
            return None, 'exact_output_columns_mismatch'
        projections: List[str] = []
        for canonical_name, result_label in zip(query.canonical_output_columns, query.result_columns):
            physical = obj.canonical_to_physical.get(canonical_name)
            if not physical:
                return None, 'exact_output_columns_mismatch'
            quoted_physical = self._quote_identifier(physical)
            if physical.lower() == result_label.lower():
                projections.append(quoted_physical)
            else:
                projections.append(f"{quoted_physical} AS {self._quote_identifier(result_label)}")
        sql = f"SELECT {', '.join(projections)} FROM derived.{obj.name}"
        order_sql = self._build_order_by_sql(query)
        if query.order_by and order_sql is None:
            return None, 'order_by_not_reconstructible'
        if order_sql:
            sql += f" ORDER BY {order_sql}"
        if query.limit is not None:
            sql += f" LIMIT {query.limit}"
        return {"sql": sql, "reason": "exact derived object match"}, None

    def _build_order_by_sql(self, query: QuerySpec) -> Optional[str]:
        if not query.order_by:
            return ''
        parts: List[str] = []
        for item in query.order_by:
            canonical = str(item.get('canonical_output') or '').lower()
            result_label = str(item.get('result_label') or '').lower()
            direction = str(item.get('direction') or 'asc').upper()
            if canonical and canonical in query.canonical_output_columns:
                parts.append(f"{self._quote_identifier(result_label)} {direction}")
            else:
                return None
        return ', '.join(parts)

    def _parse_group_by(self, clause: str, aliases: Sequence[str]) -> List[str]:
        clause = clause.strip()
        if not clause:
            return []
        items = [item.strip() for item in clause.split(',') if item.strip()]
        result: List[str] = []
        for item in items:
            if item.isdigit():
                idx = int(item) - 1
                if 0 <= idx < len(aliases):
                    result.append(aliases[idx])
            else:
                result.append(item.strip('"'))
        return result

