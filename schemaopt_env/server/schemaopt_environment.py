
"""Core schema optimization environment implementation backed by real DuckDB workloads."""

from __future__ import annotations

from collections import Counter
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
    from ..models import SchemaOptAction, SchemaOptObservation, SchemaOptState
    from ..spider_registry import list_spider_database_summaries
    from ..models import QuerySpec, TaskSpec
    from ..tasks import build_spider_episode_task, cluster_lookup, match_queries, query_lookup, similar_query_ids, visible_queries_for_cluster
except ImportError:
    from models import SchemaOptAction, SchemaOptObservation, SchemaOptState
    from spider_registry import list_spider_database_summaries
    from models import QuerySpec, TaskSpec
    from tasks import build_spider_episode_task, cluster_lookup, match_queries, query_lookup, similar_query_ids, visible_queries_for_cluster

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
            pass
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
    set_operations: List[str] = field(default_factory=list)

@dataclass
class DerivedObject:
    name: str
    object_kind: str
    sql_definition: str
    source_objects: List[str]
    grain_dims: List[str]
    available_columns: List[str]
    column_types: Dict[str, str]
    parsed_sql: ParsedSQL
    row_count: int
    storage_bytes_estimate: int
    build_runtime_ms: float
    signature: str
    used_by_visible_queries: set[str] = field(default_factory=set)
    used_by_clusters: set[str] = field(default_factory=set)
    def summary(self) -> Dict[str, Any]:
        return {"name": self.name, "object_kind": self.object_kind, "source_objects": list(self.source_objects), "grain_dims": list(self.grain_dims), "available_columns": list(self.available_columns), "row_count": self.row_count, "storage_bytes_estimate": self.storage_bytes_estimate, "build_runtime_ms": round(self.build_runtime_ms, 4), "signature": self.signature, "used_by_visible_queries": sorted(self.used_by_visible_queries), "used_by_clusters": sorted(self.used_by_clusters)}

class SchemaOptEnvironment(Environment[SchemaOptAction, SchemaOptObservation, SchemaOptState]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    LAST_GRADER_REPORT: Dict[str, Any] = {"available": False, "reason": "No episode executed yet."}

    def __init__(self):
        self._task: Optional[TaskSpec] = None
        self._visible_query_lookup: Dict[str, QuerySpec] = {}
        self._all_query_lookup: Dict[str, QuerySpec] = {}
        self._cluster_lookup: Dict[str, Any] = {}
        self._derived_objects: Dict[str, DerivedObject] = {}
        self._checkpoints: List[Dict[str, Any]] = []
        self._retrieval_context: Dict[str, Any] = {}
        self._benchmark_context: Dict[str, Any] = {}
        self._router_summary: Dict[str, Any] = {}
        self._last_feedback: Dict[str, Any] = {}
        self._base_table_stats: Dict[str, Dict[str, Any]] = {}
        self._baseline_cache: Dict[str, QueryExecution] = {}
        self._evaluation_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._latest_visible_benchmark: Dict[str, float] = {"gated_improvement": 0.0, "correctness_coverage": 1.0}
        self._state = SchemaOptState(episode_id=str(uuid4()), step_count=0)
        self._episode_root: Optional[Path] = None
        self._episode_db_path: Optional[Path] = None
        self._spider_mode: bool = False
        self._con: Any = None
        self._duckdb: Any = None
        self._sqlglot: Any = None

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_id: Optional[str] = None, **kwargs: Any) -> SchemaOptObservation:
        db_id = kwargs.get("db_id")
        spider_split = kwargs.get("spider_split") or kwargs.get("split") or "dev"
        spider_sample_size = int(kwargs.get("spider_sample_size") or 24)
        spider_holdout_size = int(kwargs.get("spider_holdout_size") or 8)
        spider_seed = kwargs.get("spider_seed", seed)
        spider_min_complexity = float(kwargs.get("spider_min_complexity") or 0.0)

        selected_db_id = db_id or task_id or kwargs.get("task_id")
        if not selected_db_id:
            spider_dbs = list_spider_database_summaries()
            selected_db_id = spider_dbs[0]["db_id"] if spider_dbs else None
        if not selected_db_id:
            raise RuntimeError("No Spider databases are available")

        self._spider_mode = True
        self._task = build_spider_episode_task(
            db_id=selected_db_id,
            split=spider_split,
            sample_size=spider_sample_size,
            holdout_size=spider_holdout_size,
            seed=spider_seed,
            min_complexity=spider_min_complexity,
        )
        selected_task = self._task.task_id
        self._derived_objects = {}
        self._checkpoints = []
        self._retrieval_context = {"last_request": None, "matched_queries": [], "matched_clusters": [], "retrieval_count": 0}
        self._benchmark_context = {"baseline_weighted_cost": None, "current_weighted_cost": None, "raw_improvement": 0.0, "gated_improvement": 0.0, "correctness_coverage": 1.0, "routed_query_count": 0, "incorrect_query_count": 0, "last_benchmarked_query_ids": [], "last_benchmarked_cluster_id": None, "latest_plan_deltas": {}}
        self._last_feedback = {
            "event": "reset",
            "task_id": selected_task,
            "spider_mode": self._spider_mode,
            "db_id": self._task.domain if self._spider_mode else None,
            "split": self._task.engine_capabilities.get("split") if self._spider_mode else None,
        }
        self._baseline_cache = {}
        self._evaluation_cache = {}
        self._latest_visible_benchmark = {"gated_improvement": 0.0, "correctness_coverage": 1.0}
        self._bootstrap_episode_database()
        if self._spider_mode:
            filtered = self._filter_spider_queries(self._task)
            dropped_visible = len(self._task.visible_queries) - len(filtered.visible_queries)
            dropped_holdout = len(self._task.holdout_queries) - len(filtered.holdout_queries)
            self._task = filtered
            self._last_feedback["dropped_non_executable_visible_queries"] = dropped_visible
            self._last_feedback["dropped_non_executable_holdout_queries"] = dropped_holdout

        self._visible_query_lookup = {q.query_id: q for q in self._task.visible_queries}
        self._all_query_lookup = query_lookup(self._task)
        self._cluster_lookup = cluster_lookup(self._task)
        self._router_summary = {"queries_routed": 0, "queries_unrouted": len(self._task.visible_queries), "dominant_rejection_reason": None, "candidate_object_coverage": {}, "last_scope": "reset"}
        self._base_table_stats = {table.name: self._collect_table_stats(table.name) for table in self._task.tables}
        self._state = SchemaOptState(episode_id=str(uuid4()) if episode_id is None else episode_id, step_count=0, done=False, task_id=self._task.task_id, difficulty=self._task.difficulty, derived_object_count=0, checkpoint_count=0, retrieval_count=0, benchmark_runs=0, storage_used_multiplier=0.0, final_score=None, last_error=None)
        payload = self._task.reset_payload()
        return SchemaOptObservation(status="ok", message=f"Initialized task {self._task.task_id}", catalog_summary=self._catalog_summary(payload), workload_summary=payload["workload_summary"], retrieval_context=self._retrieval_context, benchmark_context=self._benchmark_context, router_summary=self._router_summary, action_feedback=self._last_feedback, reward=0.0, done=False, metadata={"task": payload["task"], "spider_mode": self._spider_mode})

    @property
    def state(self) -> SchemaOptState:
        return self._state

    @classmethod
    def latest_report(cls) -> Dict[str, Any]:
        return cls.LAST_GRADER_REPORT

    @classmethod
    def run_baseline(cls) -> Dict[str, Any]:
        env = cls()
        spider_dbs = list_spider_database_summaries()
        if not spider_dbs:
            raise RuntimeError("No Spider databases are available")
        env.reset(db_id=spider_dbs[0]["db_id"])
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

    def _import_sqlglot(self):
        if self._sqlglot is not None:
            return self._sqlglot
        try:
            import sqlglot  # type: ignore
            from sqlglot import exp  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sqlglot is required for AST-based SQL metadata parsing") from exc
        self._sqlglot = (sqlglot, exp)
        return self._sqlglot

    def _bootstrap_episode_database(self) -> None:
        duckdb = self._import_duckdb()
        if self._con is not None:
            self._con.close()
        if self._episode_root and self._episode_root.exists():
            shutil.rmtree(self._episode_root, ignore_errors=True)
        self._episode_root = Path(tempfile.mkdtemp(prefix="schemaopt_episodes_")) / str(uuid4())
        self._episode_root.mkdir(parents=True, exist_ok=True)
        source_db_path = Path(self._task.database_path)
        if not source_db_path.exists():
            raise FileNotFoundError(f"Task database not found: {source_db_path}")
        if self._spider_mode:
            self._episode_db_path = self._episode_root / "spider_episode.duckdb"
            self._con = duckdb.connect(str(self._episode_db_path))
            self._hydrate_spider_sqlite(source_db_path)
        else:
            self._episode_db_path = self._episode_root / source_db_path.name
            shutil.copy2(source_db_path, self._episode_db_path)
            self._con = duckdb.connect(str(self._episode_db_path))
        self._con.execute("CREATE SCHEMA IF NOT EXISTS derived")

    def _hydrate_spider_sqlite(self, sqlite_path: Path) -> None:
        sqlite_file = str(sqlite_path).replace("'", "''")
        alias = "spider_src"
        self._con.execute("INSTALL sqlite")
        self._con.execute("LOAD sqlite")
        self._con.execute("SET sqlite_all_varchar=true")
        self._con.execute(f"ATTACH '{sqlite_file}' AS {alias} (TYPE SQLITE)")
        table_rows = self._con.execute(
            f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_catalog = '{alias}'
              AND table_schema = 'main'
              AND table_name NOT LIKE 'sqlite_%'
            ORDER BY table_name
            """
        ).fetchall()
        for row in table_rows:
            table_name = str(row[0])
            quoted = '"' + table_name.replace('"', '""') + '"'
            self._con.execute(f"CREATE OR REPLACE TABLE {quoted} AS SELECT * FROM {alias}.main.{quoted}")

    def _filter_spider_queries(self, task: TaskSpec) -> TaskSpec:
        executable_visible: List[QuerySpec] = []
        executable_holdout: List[QuerySpec] = []

        for query in task.visible_queries:
            try:
                self._execute_query(query.sql, repeats=1, warmup=0)
                executable_visible.append(query)
            except Exception:
                continue
        for query in task.holdout_queries:
            try:
                self._execute_query(query.sql, repeats=1, warmup=0)
                executable_holdout.append(query)
            except Exception:
                continue

        if not executable_visible:
            raise RuntimeError("Spider workload sampling produced no DuckDB-executable visible queries")

        visible_lookup = {query.query_id: query for query in executable_visible}
        clusters = []
        for cluster in task.clusters:
            ids = tuple(query_id for query_id in cluster.query_ids if query_id in visible_lookup)
            if not ids:
                continue
            freq = sum(visible_lookup[query_id].frequency_weight for query_id in ids)
            weighted = sum(visible_lookup[query_id].weighted_cost for query_id in ids)
            clusters.append(
                cluster.__class__(
                    cluster_id=cluster.cluster_id,
                    label=cluster.label,
                    business_label=cluster.business_label,
                    query_ids=ids,
                    query_count=len(ids),
                    total_frequency_weight=round(freq, 6),
                    total_weighted_baseline_cost=round(weighted, 6),
                    top_tables=cluster.top_tables,
                    common_operator_patterns=cluster.common_operator_patterns,
                    representative_dimensions=cluster.representative_dimensions,
                    representative_measures=cluster.representative_measures,
                    hotspot_rank=cluster.hotspot_rank,
                    preferred_object_kind=cluster.preferred_object_kind,
                    representative_query_id=ids[0],
                    cluster_grain_emphasis=cluster.cluster_grain_emphasis,
                    suggested_exact_derived_shape=cluster.suggested_exact_derived_shape,
                    reference_rewrite_feasible=cluster.reference_rewrite_feasible,
                )
            )
        if not clusters:
            primary = executable_visible[0]
            clusters = [
                task.clusters[0].__class__(
                    cluster_id=f"spider_{task.domain}_filtered",
                    label=f"Spider filtered workload for {task.domain}",
                    business_label=task.domain,
                    query_ids=tuple(query.query_id for query in executable_visible),
                    query_count=len(executable_visible),
                    total_frequency_weight=round(sum(query.frequency_weight for query in executable_visible), 6),
                    total_weighted_baseline_cost=round(sum(query.weighted_cost for query in executable_visible), 6),
                    top_tables=tuple(primary.tables[:4]),
                    common_operator_patterns=tuple(sorted({pattern for query in executable_visible for pattern in query.plan_features})),
                    representative_dimensions=(),
                    representative_measures=(),
                    hotspot_rank=1,
                    preferred_object_kind="agg_matview",
                    representative_query_id=primary.query_id,
                    cluster_grain_emphasis=(),
                    suggested_exact_derived_shape={
                        "object_kind": "agg_matview",
                        "source_objects": list(primary.tables),
                        "group_by": [],
                        "canonical_predicates": [],
                        "measure_columns": [],
                        "aggregate_functions": [],
                    },
                    reference_rewrite_feasible=True,
                )
            ]

        return task.__class__(
            task_id=task.task_id,
            difficulty=task.difficulty,
            domain=task.domain,
            objective=task.objective,
            seed_source=task.seed_source,
            dataset_dir=task.dataset_dir,
            database_path=task.database_path,
            tables=task.tables,
            visible_queries=tuple(executable_visible),
            holdout_queries=tuple(executable_holdout),
            clusters=tuple(clusters),
            budgets=task.budgets,
            allowed_object_kinds=task.allowed_object_kinds,
            engine_capabilities=task.engine_capabilities,
        )

    def _catalog_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        catalog = payload["catalog_summary"]
        catalog["tables"] = [self._base_table_stats.get(table.name, table.to_dict()) for table in self._task.tables]
        catalog["derived_objects"] = [obj.summary() for obj in self._derived_objects.values()]
        catalog["storage_usage_estimate"] = sum(obj.storage_bytes_estimate for obj in self._derived_objects.values())
        catalog["refresh_cost_estimate"] = round(sum(obj.build_runtime_ms for obj in self._derived_objects.values()), 4)
        return catalog

    def _build_observation(self, status: str, message: str, reward: float, done: bool, feedback: Dict[str, Any]) -> SchemaOptObservation:
        payload = self._task.reset_payload()
        return SchemaOptObservation(status=status, message=message, catalog_summary=self._catalog_summary(payload), workload_summary=payload["workload_summary"], retrieval_context=self._retrieval_context, benchmark_context=self._benchmark_context, router_summary=self._router_summary, action_feedback=feedback, reward=round(reward, 6), done=done, metadata={"task": payload["task"]})
    def step(self, action: SchemaOptAction, timeout_s: Optional[float] = None, **kwargs: Any) -> SchemaOptObservation:
        self._state.step_count += 1
        reward = 0.0
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
            elif action.operation == "inspect_cluster":
                cluster = self._cluster_lookup[action.target_id or ""]
                feedback = {"event": "inspect_cluster", "cluster": cluster.to_summary(), "example_query_ids": list(cluster.query_ids[:3])}
                message = f"Cluster summary returned for {action.target_id}."
            elif action.operation == "inspect_query":
                query = self._get_visible_query(action.target_id or "")
                feedback = {"event": "inspect_query", "query": query.summary()}
                message = f"Query summary returned for {action.target_id}."
            elif action.operation == "inspect_query_plan":
                feedback = self._inspect_query_plan(action.target_id or "")
                message = f"Plan summary returned for {action.target_id}."
            elif action.operation == "inspect_router_status":
                if action.query_ids:
                    query_ids = list(action.query_ids)
                elif action.cluster_id:
                    query_ids = [query.query_id for query in visible_queries_for_cluster(self._task, action.cluster_id)]
                else:
                    query_ids = list(self._visible_query_lookup.keys())
                feedback = self._inspect_router_status(query_ids, action.cluster_id)
                message = "Router status returned."
            elif action.operation == "retrieve_queries":
                feedback = self._retrieve_queries(action)
                self._state.retrieval_count += 1
                message = f"Retrieved {len(feedback['matched_queries'])} query summaries."
            elif action.operation == "get_query_context":
                feedback = self._get_query_context(action.query_ids)
                self._state.retrieval_count += 1
                message = f"Returned full context for {len(action.query_ids)} queries."
            elif action.operation == "create_derived_object":
                feedback = self._upsert_derived_object(action, False)
                reward += feedback.get("reward_delta", 0.0)
                message = feedback.get("message", f"Created derived object {action.name}.")
            elif action.operation == "modify_derived_object":
                feedback = self._upsert_derived_object(action, True)
                reward += feedback.get("reward_delta", 0.0)
                message = feedback.get("message", f"Modified derived object {action.name}.")
            elif action.operation == "drop_derived_object":
                removed = self._derived_objects.pop(action.target_id or "")
                self._con.execute(f"DROP TABLE IF EXISTS derived.{removed.name}")
                self._evaluation_cache = {}
                feedback = {"event": "drop_derived_object", "removed": removed.summary()}
                reward -= 0.002
                message = f"Dropped derived object {action.target_id}."
            elif action.operation == "list_derived_objects":
                feedback = {"event": "list_derived_objects", "derived_objects": [obj.summary() for obj in self._derived_objects.values()]}
                message = "Derived object list returned."
            elif action.operation == "checkpoint":
                label = action.name or f"checkpoint_{len(self._checkpoints)+1}"
                self._checkpoints.append({"label": label, "derived_objects": copy.deepcopy(self._derived_objects)})
                feedback = {"event": "checkpoint", "label": label, "checkpoint_count": len(self._checkpoints)}
                message = "Checkpoint created."
            elif action.operation == "revert_checkpoint":
                snapshot = next((item for item in reversed(self._checkpoints) if action.target_id is None or item["label"] == action.target_id), None)
                if snapshot is None:
                    raise ValueError("Unknown checkpoint label")
                for name in list(self._derived_objects.keys()):
                    self._con.execute(f"DROP TABLE IF EXISTS derived.{name}")
                self._derived_objects = {}
                for obj in copy.deepcopy(snapshot["derived_objects"]).values():
                    self._con.execute(f"CREATE TABLE derived.{obj.name} AS ({obj.sql_definition})")
                    self._derived_objects[obj.name] = obj
                self._evaluation_cache = {}
                feedback = {"event": "revert_checkpoint", "label": snapshot["label"], "restored_objects": [obj.summary() for obj in self._derived_objects.values()]}
                message = "Checkpoint restored."
            elif action.operation == "benchmark_subset":
                feedback = self._benchmark_action([self._get_visible_query(query_id) for query_id in action.query_ids], None)
                reward += feedback["reward_delta"]
                message = f"Benchmarked {len(action.query_ids)} visible queries."
            elif action.operation == "benchmark_cluster":
                feedback = self._benchmark_action(visible_queries_for_cluster(self._task, action.cluster_id or ""), action.cluster_id)
                reward += feedback["reward_delta"]
                message = f"Benchmarked cluster {action.cluster_id}."
            elif action.operation == "submit":
                feedback = self._submit_episode()
                reward += feedback["terminal_reward"]
                status = "completed"
                done = True
                message = "Final benchmark submitted."
            else:
                raise ValueError(f"Unsupported action: {action.operation}")
        except Exception as exc:
            status = "error"
            message = f"ERROR: {exc}"
            feedback = {"event": action.operation, "error": str(exc)}
            reward -= 0.05
            self._state.last_error = str(exc)
        self._last_feedback = feedback
        self._state.done = done
        self._state.derived_object_count = len(self._derived_objects)
        self._state.checkpoint_count = len(self._checkpoints)
        self._state.storage_used_multiplier = round(sum(obj.storage_bytes_estimate for obj in self._derived_objects.values()) / max(1.0, float(self._task.budgets.get("max_storage_bytes", 1.0))), 6)
        if done:
            self._state.final_score = feedback.get("final_score")
        return self._build_observation(status, message, reward, done, feedback)
    
    def _inspect_query_plan(self, query_id: str) -> Dict[str, Any]:
        query = self._get_visible_query(query_id)
        baseline = self._baseline_for_query(query)
        decision = self._evaluate_query(query, False)
        route_summary = self._route_summary(query_id, decision)
        self._router_summary = self._summarize_routes([route_summary], f"query:{query_id}")
        return {
            "event": "inspect_query_plan",
            "query_id": query_id,
            "baseline_plan": baseline.plan.summary(),
            "current_plan": route_summary,
            "top_rewrite_candidate": decision.get("top_rewrite_candidate"),
            "dominant_rejection_reason": route_summary.get("top_rejection_reason"),
        }

    def _inspect_router_status(self, query_ids: Sequence[str], cluster_id: Optional[str]) -> Dict[str, Any]:
        routes = [self._route_summary(query_id, self._evaluate_query(self._get_visible_query(query_id), False)) for query_id in query_ids]
        self._router_summary = self._summarize_routes(routes, f"cluster:{cluster_id}" if cluster_id else "explicit_query_set")
        return {"event": "inspect_router_status", "routes": routes, "router_summary": self._router_summary}

    def _retrieve_queries(self, action: SchemaOptAction) -> Dict[str, Any]:
        mode = self._resolve_retrieval_mode(action)
        matches = match_queries(self._task, mode=mode, pattern=action.pattern, cluster_id=action.cluster_id, tables=action.tables, columns=action.columns, plan_features=action.plan_features, top_k=action.top_k)
        self._retrieval_context = {"last_request": {"mode": mode, "pattern": action.pattern, "cluster_id": action.cluster_id, "tables": list(action.tables), "columns": list(action.columns), "plan_features": list(action.plan_features), "top_k": action.top_k}, "matched_queries": [query.summary() for query in matches], "matched_clusters": sorted({query.cluster_id for query in matches}), "retrieval_count": self._state.retrieval_count + 1}
        return {"event": "retrieve_queries", **self._retrieval_context}

    def _get_query_context(self, query_ids: Sequence[str]) -> Dict[str, Any]:
        contexts = [self._get_visible_query(query_id).context(similar_query_ids(self._task, query_id)) for query_id in query_ids]
        self._retrieval_context = {"last_request": {"mode": "get_query_context", "query_ids": list(query_ids)}, "matched_queries": [ctx["query_id"] for ctx in contexts], "matched_clusters": sorted({ctx["cluster_id"] for ctx in contexts}), "retrieval_count": self._state.retrieval_count + 1}
        return {"event": "get_query_context", "query_context": contexts}

    def _upsert_derived_object(self, action: SchemaOptAction, modify: bool) -> Dict[str, Any]:
        name = (action.name or "").strip()
        if not _IDENTIFIER_RE.match(name):
            raise ValueError(f"Invalid derived object name: {name}")
        if modify and name not in self._derived_objects:
            raise ValueError(f"Derived object '{name}' does not exist")
        if not modify and name in self._derived_objects:
            raise ValueError(f"Derived object '{name}' already exists")
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
                "reward_delta": -0.02,
            }
        if modify:
            self._con.execute(f"DROP TABLE IF EXISTS derived.{name}")
        start = time.perf_counter()
        self._con.execute(f"CREATE TABLE derived.{name} AS ({action.sql_definition})")
        build_runtime_ms = (time.perf_counter() - start) * 1000.0
        describe_rows = self._con.execute(f"DESCRIBE derived.{name}").fetchall()
        column_types = {str(row[0]).lower(): str(row[1]) for row in describe_rows}
        available_columns = list(column_types.keys())
        row_count = int(self._con.execute(f"SELECT COUNT(*) FROM derived.{name}").fetchone()[0])
        obj = DerivedObject(name=name, object_kind=action.object_kind or "agg_matview", sql_definition=action.sql_definition or "", source_objects=source_objects, grain_dims=grain_dims, available_columns=available_columns, column_types=column_types, parsed_sql=parsed, row_count=row_count, storage_bytes_estimate=self._estimate_storage_bytes(row_count, column_types), build_runtime_ms=build_runtime_ms, signature=signature)
        self._derived_objects[name] = obj
        self._evaluation_cache = {}
        diagnostics = self._derived_object_diagnostics(obj, action.intended_clusters)
        reward_delta = 0.01 if diagnostics["eligible_visible_queries"] else 0.0
        if diagnostics["is_empty_object"]:
            reward_delta = -0.02
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
            "reward_delta": reward_delta,
            **diagnostics,
        }

    def _benchmark_action(self, queries: Sequence[QuerySpec], cluster_id: Optional[str]) -> Dict[str, Any]:
        summary = self._benchmark_queries(queries, True)
        prev_imp = self._latest_visible_benchmark.get("gated_improvement", 0.0)
        prev_corr = self._latest_visible_benchmark.get("correctness_coverage", 1.0)
        improvement_delta = summary["gated_improvement"] - prev_imp
        correctness_delta = summary["correctness_coverage"] - prev_corr
        reward_delta = improvement_delta + 0.05 * correctness_delta - summary["budget_penalty"]
        if abs(improvement_delta) < 1e-9 and summary["incorrect_query_count"] == 0 and summary["budget_penalty"] == 0:
            reward_delta = 0.0
        reward_delta = max(-0.25, min(0.25, reward_delta))
        self._latest_visible_benchmark = {"gated_improvement": summary["gated_improvement"], "correctness_coverage": summary["correctness_coverage"]}
        self._benchmark_context = {"baseline_weighted_cost": summary["baseline_weighted_cost"], "current_weighted_cost": summary["actual_current_weighted_cost"], "raw_improvement": summary["raw_improvement"], "gated_improvement": summary["gated_improvement"], "correctness_coverage": summary["correctness_coverage"], "routed_query_count": summary["routed_query_count"], "incorrect_query_count": summary["incorrect_query_count"], "last_benchmarked_query_ids": [query.query_id for query in queries], "last_benchmarked_cluster_id": cluster_id, "latest_plan_deltas": summary["plan_deltas"]}
        self._router_summary = self._summarize_routes(summary["per_query"], f"cluster:{cluster_id}" if cluster_id else "subset")
        self._state.benchmark_runs += 1
        summary["event"] = "benchmark_cluster" if cluster_id else "benchmark_subset"
        summary["reward_delta"] = round(reward_delta, 6)
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
        SchemaOptEnvironment.LAST_GRADER_REPORT = {"available": True, "task_id": self._task.task_id, "episode_id": self._state.episode_id, "score": final_score, "visible_summary": visible, "holdout_summary": holdout, "migration_score": migration, "storage_score": storage}
        return {"event": "submit", "final_score": final_score, "visible_summary": visible, "holdout_summary": holdout, "migration_score": migration, "storage_score": storage, "terminal_reward": final_score}

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
            try:
                current = self._execute_query(rewrite["sql"])
            except Exception as exc:
                best["rejection_reasons"].append({"object_name": obj.name, "reason": "rewrite_execution_error", "detail": str(exc)[:180]})
                continue
            correctness = self._compare_results(baseline, current, query.sql)
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
        if self._spider_mode:
            return self._build_spider_rewrite(query, obj)
        if sorted(query.canonical_tables) != sorted(obj.parsed_sql.canonical_tables):
            return None, "table_mismatch"
        if set(query.canonical_filter_predicates) != set(obj.parsed_sql.canonical_filter_predicates):
            return None, "predicate_mismatch"
        query_dims = [item.lower() for item in query.group_by]
        if not set(query_dims).issubset(set(obj.parsed_sql.group_by)):
            return None, "group_by_not_subset"
        if not set(query.measure_columns).issubset(set(obj.available_columns)):
            return None, "measure_columns_missing"
        if query_dims == obj.parsed_sql.group_by and set(query.columns).issubset(set(obj.available_columns)):
            selected_columns = list(query.columns) if query.columns else list(obj.available_columns)
            if not selected_columns:
                return {"sql": f"SELECT * FROM derived.{obj.name}", "reason": "exact derived object match"}, None
            select_expr = ", ".join(self._quote_identifier(column) for column in selected_columns)
            return {"sql": f"SELECT {select_expr} FROM derived.{obj.name}", "reason": "exact derived object match"}, None
        if obj.object_kind not in {"agg_matview", "denorm_table", "join_matview"}:
            return None, "object_kind_not_rollup_compatible"
        if not all(func in {"count", "sum"} for func in query.aggregate_functions):
            return None, "aggregate_functions_not_supported"
        select_cols = list(query_dims) + [f"SUM({measure}) AS {measure}" for measure in query.measure_columns]
        sql = f"SELECT {', '.join(select_cols)} FROM derived.{obj.name}"
        if query_dims:
            sql += " GROUP BY " + ", ".join(query_dims)
        return {"sql": sql, "reason": "rollup over derived object"}, None

    def _build_spider_rewrite(self, query: QuerySpec, obj: DerivedObject) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        try:
            query_parsed = self._parse_sql_metadata(query.sql)
        except Exception:
            return None, "query_parse_failed"

        diagnostics = self._spider_fragment_diagnostics(query_parsed, obj.parsed_sql)
        if not diagnostics["eligible"]:
            return None, diagnostics["reason"]

        if diagnostics["exact_signature_match"]:
            return {
                "sql": f"SELECT * FROM derived.{obj.name}",
                "reason": "spider_exact_fragment_match",
            }, None

        if diagnostics.get("partial_predicate_refinement_allowed"):
            extra_predicates = diagnostics.get("extra_predicates") or []
            sql = f"SELECT * FROM derived.{obj.name}"
            if extra_predicates:
                sql += " WHERE " + " AND ".join(extra_predicates)
            return {
                "sql": sql,
                "reason": "spider_partial_predicate_refinement",
            }, None

        return None, diagnostics.get("reason") or "partial_fragment_not_enabled"

    def _spider_fragment_diagnostics(self, query_parsed: ParsedSQL, object_parsed: ParsedSQL) -> Dict[str, Any]:
        query_tables = set(query_parsed.canonical_tables)
        object_tables = set(object_parsed.canonical_tables)
        shared_tables = query_tables & object_tables
        table_overlap_ratio = 0.0 if not query_tables else len(shared_tables) / max(1, len(query_tables))

        query_preds = set(query_parsed.canonical_filter_predicates)
        object_preds = set(object_parsed.canonical_filter_predicates)
        shared_preds = query_preds & object_preds
        predicate_overlap_ratio = 0.0 if not query_preds else len(shared_preds) / max(1, len(query_preds))

        exact_signature_match = (
            query_tables == object_tables
            and query_preds == object_preds
            and set(query_parsed.group_by) == set(object_parsed.group_by)
            and set(query_parsed.aggregate_functions) == set(object_parsed.aggregate_functions)
            and set(query_parsed.set_operations) == set(object_parsed.set_operations)
        )

        if not shared_tables:
            return {
                "eligible": False,
                "reason": "no_table_overlap",
                "table_overlap_ratio": table_overlap_ratio,
                "predicate_overlap_ratio": predicate_overlap_ratio,
                "exact_signature_match": exact_signature_match,
            }

        if query_parsed.set_operations and not exact_signature_match:
            return {
                "eligible": False,
                "reason": "set_operation_requires_exact_match",
                "table_overlap_ratio": table_overlap_ratio,
                "predicate_overlap_ratio": predicate_overlap_ratio,
                "exact_signature_match": exact_signature_match,
            }

        query_group = set(query_parsed.group_by)
        object_group = set(object_parsed.group_by)
        query_aggs = set(query_parsed.aggregate_functions)
        object_aggs = set(object_parsed.aggregate_functions)

        if query_group != object_group:
            return {
                "eligible": False,
                "reason": "group_by_incompatible",
                "table_overlap_ratio": table_overlap_ratio,
                "predicate_overlap_ratio": predicate_overlap_ratio,
                "exact_signature_match": exact_signature_match,
            }
        if query_aggs != object_aggs:
            return {
                "eligible": False,
                "reason": "aggregate_incompatible",
                "table_overlap_ratio": table_overlap_ratio,
                "predicate_overlap_ratio": predicate_overlap_ratio,
                "exact_signature_match": exact_signature_match,
            }

        object_pred_list = list(object_parsed.canonical_filter_predicates)
        query_pred_list = list(query_parsed.canonical_filter_predicates)
        object_pred_set = set(object_pred_list)
        query_pred_set = set(query_pred_list)

        if not object_pred_set.issubset(query_pred_set):
            return {
                "eligible": False,
                "reason": "object_predicates_not_subset",
                "table_overlap_ratio": table_overlap_ratio,
                "predicate_overlap_ratio": predicate_overlap_ratio,
                "exact_signature_match": exact_signature_match,
            }

        extra_predicates = [pred for pred in query_pred_list if pred not in object_pred_set]
        if extra_predicates:
            object_columns = {column.lower() for column in object_parsed.projection_aliases + object_parsed.group_by + object_parsed.measure_columns}
            for predicate in extra_predicates:
                referenced = self._extract_predicate_columns(predicate)
                if not referenced.issubset(object_columns):
                    return {
                        "eligible": False,
                        "reason": "partial_predicate_columns_missing",
                        "table_overlap_ratio": table_overlap_ratio,
                        "predicate_overlap_ratio": predicate_overlap_ratio,
                        "exact_signature_match": exact_signature_match,
                        "extra_predicates": extra_predicates,
                    }

        return {
            "eligible": True,
            "reason": "eligible_for_spider_fragment",
            "table_overlap_ratio": table_overlap_ratio,
            "predicate_overlap_ratio": predicate_overlap_ratio,
            "exact_signature_match": exact_signature_match,
            "partial_predicate_refinement_allowed": bool(extra_predicates) and not query_parsed.set_operations and not object_parsed.set_operations,
            "extra_predicates": extra_predicates,
        }

    def _quote_identifier(self, identifier: str) -> str:
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def _extract_predicate_columns(self, predicate_sql: str) -> set[str]:
        try:
            sqlglot, exp = self._import_sqlglot()
            expression = sqlglot.parse_one(f"SELECT 1 WHERE {predicate_sql}", read="sqlite")
        except Exception:
            return set()
        columns: set[str] = set()
        for column in expression.find_all(exp.Column):
            name = str(getattr(column, "name", "") or "").strip().strip('"').lower()
            if name:
                columns.add(name)
        return columns

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
    def _compare_results(self, baseline: QueryExecution, current: QueryExecution, query_sql: Optional[str] = None) -> bool:
        if baseline.column_names != current.column_names or len(baseline.rows) != len(current.rows):
            return False
        left = [self._normalize_row(row) for row in baseline.rows]
        right = [self._normalize_row(row) for row in current.rows]
        if not self._is_order_sensitive_sql(query_sql):
            left = sorted(left)
            right = sorted(right)
        return left == right

    def _is_order_sensitive_sql(self, sql: Optional[str]) -> bool:
        if not sql:
            return False
        lowered = " ".join(sql.lower().split())
        return (" order by " in f" {lowered} ") or (" limit " in f" {lowered} ")

    def _normalize_row(self, row: Tuple[Any, ...]) -> Tuple[Any, ...]:
        result: List[Any] = []
        for value in row:
            if isinstance(value, float):
                result.append(round(value, 9))
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
            "set_operations": sorted(parsed.set_operations),
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
        sqlglot, exp = self._import_sqlglot()
        normalized = sql.strip().rstrip(";")
        if not normalized:
            raise ValueError("SQL definition cannot be empty")

        try:
            parsed = sqlglot.parse_one(normalized, read="sqlite")
        except Exception as exc:
            raise ValueError(f"Failed to parse SQL definition: {exc}") from exc

        select_nodes = list(parsed.find_all(exp.Select))
        if not select_nodes:
            raise ValueError("Only SELECT-derived objects are supported")
        root_select = select_nodes[0]

        table_names: List[str] = []
        for table in parsed.find_all(exp.Table):
            base_name = str(getattr(table, "name", "") or "").strip()
            if not base_name:
                continue
            db_name = str(getattr(table, "db", "") or "").strip()
            if db_name:
                table_names.append(f"{db_name}.{base_name}")
            else:
                table_names.append(base_name)
        canonical_tables = [self._canonicalize_table_name(item) for item in table_names]
        table_names = self._dedupe_preserve_order(table_names)
        canonical_tables = self._dedupe_preserve_order(canonical_tables)

        projection_aliases: List[str] = []
        measure_columns: List[str] = []
        aggregate_functions: List[str] = []
        for projection in list(root_select.expressions or []):
            alias_name = (projection.alias_or_name or projection.output_name or projection.sql(dialect="sqlite")).strip('"').lower()
            if alias_name:
                projection_aliases.append(alias_name)
            agg_nodes = list(projection.find_all(exp.AggFunc))
            if agg_nodes:
                if alias_name:
                    measure_columns.append(alias_name)
                else:
                    measure_columns.append(projection.sql(dialect="sqlite").lower())
                for agg in agg_nodes:
                    aggregate_functions.append(str(getattr(agg, "key", "agg")).lower())

        group_by: List[str] = []
        group_expr = root_select.args.get("group")
        if group_expr is not None:
            for item in list(group_expr.expressions or []):
                if isinstance(item, exp.Literal) and item.is_int:
                    idx = int(item.this) - 1
                    if 0 <= idx < len(projection_aliases):
                        group_by.append(projection_aliases[idx])
                else:
                    group_by.append((item.alias_or_name or item.output_name or item.sql(dialect="sqlite")).strip('"').lower())

        raw_filter_predicates: List[str] = []
        where_expr = root_select.args.get("where")
        if where_expr is not None and where_expr.this is not None:
            raw_filter_predicates = self._flatten_and_predicates(where_expr.this)

        set_operations: List[str] = []
        for op_cls, label in ((exp.Union, "union"), (exp.Intersect, "intersect"), (exp.Except, "except")):
            if any(True for _ in parsed.find_all(op_cls)):
                set_operations.append(label)

        return ParsedSQL(
            tables=table_names,
            canonical_tables=canonical_tables,
            projection_aliases=projection_aliases,
            group_by=group_by,
            raw_filter_predicates=raw_filter_predicates,
            canonical_filter_predicates=[self._normalize_predicate(item) for item in raw_filter_predicates],
            measure_columns=measure_columns,
            aggregate_functions=aggregate_functions,
            set_operations=set_operations,
        )

    def _flatten_and_predicates(self, expression: Any) -> List[str]:
        _, exp = self._import_sqlglot()
        if expression is None:
            return []
        if isinstance(expression, exp.And):
            left = self._flatten_and_predicates(expression.this)
            right = self._flatten_and_predicates(expression.expression)
            return left + right
        return [expression.sql(dialect="sqlite")]

    def _dedupe_preserve_order(self, values: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _normalize_predicate(self, predicate: str) -> str:
        normalized = " ".join(predicate.strip().lower().split())
        normalized = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.((?:"[^"]+")|(?:[a-zA-Z_][a-zA-Z0-9_]*))', lambda match: match.group(1), normalized)
        normalized = normalized.replace("\"", "")
        normalized = normalized.replace("( ", "(").replace(" )", ")")
        return normalized


