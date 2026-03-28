"""Core DataDAG environment implementation for Task 1."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    import duckdb
except ImportError as e:
    raise ImportError("duckdb is required. Install with: pip install duckdb") from e

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State

    from ..models import DatadagAction, DatadagObservation, DatadagState
except ImportError:
    from openenv_core.env_server.interfaces import Environment
    from openenv_core.env_server.types import State

    from models import DatadagAction, DatadagObservation, DatadagState


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class NodeSpec:
    sql: str
    deps: List[str]
    version: int = 1


class DatadagEnvironment(Environment[DatadagAction, DatadagObservation, DatadagState]):
    """Data engineering DAG environment with deterministic Task 1 grading."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    LAST_GRADER_REPORT: Dict[str, Any] = {
        "available": False,
        "reason": "No episode executed yet.",
    }

    def __init__(self):
        self._state = DatadagState(episode_id=str(uuid4()), step_count=0)
        self._nodes: Dict[str, NodeSpec] = {}
        self._last_error_step: Optional[int] = None
        self._last_error_signature: Optional[str] = None
        self._last_integrity: str = "valid"
        self._last_trace: str = "Environment initialized"
        self._last_task1_rubric: Optional[Dict[str, Any]] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DatadagObservation:
        self._nodes = {}
        self._last_error_step = None
        self._last_error_signature = None
        self._last_integrity = "valid"
        self._last_trace = "Task 1 initialized: build staging and mart nodes, then execute_dag."
        self._last_task1_rubric = None

        self._state = DatadagState(
            episode_id=str(uuid4()) if episode_id is None else episode_id,
            step_count=0,
            done=False,
            task_id="task1_easy",
            dag_node_count=0,
            last_error=None,
            final_score=None,
        )

        return DatadagObservation(
            dag_integrity_status="valid",
            execution_trace=self._last_trace,
            data_sample=None,
            cascading_failure_nodes=[],
            reward=0.0,
            done=False,
            metadata={
                "task": {
                    "id": "task1_easy",
                    "objective": "Create a staging model and mart model for user lifetime revenue.",
                    "required_sources": ["raw_transactions", "raw_users"],
                    "target_model": "mart_user_lifetime_revenue",
                }
            },
        )

    def step(
        self,
        action: DatadagAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DatadagObservation:
        self._state.step_count += 1
        reward = 0.0
        done = False
        data_sample = None
        cascading_failure_nodes: List[str] = []

        try:
            if action.pipeline_command == "create_node":
                trace = self._create_node(action)
                reward += 0.15
                self._last_integrity = "valid"

            elif action.pipeline_command == "update_node":
                trace = self._update_node(action)
                reward += 0.10
                if self._last_error_step is not None and self._state.step_count == self._last_error_step + 1:
                    reward += 0.20
                    trace += "\nSelf-correction bonus awarded."
                self._last_integrity = "valid"

            elif action.pipeline_command == "view_lineage":
                trace = self._view_lineage(action.model_identifier)
                reward += 0.02
                self._last_integrity = "valid"

            elif action.pipeline_command == "execute_dag":
                (
                    trace,
                    execute_ok,
                    data_sample,
                    cascading_failure_nodes,
                    terminal_score,
                    rubric_breakdown,
                ) = self._execute_dag()
                done = True
                reward += 0.20 if execute_ok else -0.20
                reward += terminal_score
                if execute_ok:
                    self._last_integrity = "valid"
                self._state.final_score = terminal_score
                self._last_task1_rubric = rubric_breakdown
                DatadagEnvironment.LAST_GRADER_REPORT = {
                    "available": True,
                    "task_id": "task1_easy",
                    "episode_id": self._state.episode_id,
                    "score": terminal_score,
                    "execute_ok": execute_ok,
                    "node_count": len(self._nodes),
                    "step_count": self._state.step_count,
                    "rubric": rubric_breakdown,
                }

            else:
                raise ValueError(f"Unsupported command: {action.pipeline_command}")

        except Exception as e:
            self._last_integrity = "compilation_error"
            self._last_error_step = self._state.step_count
            self._last_error_signature = str(e)
            self._state.last_error = str(e)
            reward -= 0.15
            trace = f"ERROR: {e}"

        self._state.done = done
        self._state.dag_node_count = len(self._nodes)
        self._last_trace = trace

        return DatadagObservation(
            dag_integrity_status=self._last_integrity,
            execution_trace=trace,
            data_sample=data_sample,
            cascading_failure_nodes=cascading_failure_nodes,
            reward=reward,
            done=done,
            metadata={
                "step": self._state.step_count,
                "node_count": len(self._nodes),
                "final_score": self._state.final_score,
                "task1_rubric": self._last_task1_rubric,
            },
        )

    @property
    def state(self) -> State:
        return self._state

    @classmethod
    def latest_report(cls) -> Dict[str, Any]:
        return cls.LAST_GRADER_REPORT

    @classmethod
    def run_baseline(cls) -> Dict[str, Any]:
        env = cls()
        env.reset()

        env.step(
            DatadagAction(
                pipeline_command="create_node",
                model_identifier="stg_transactions",
                sql_syntax=(
                    "SELECT transaction_id, "
                    "CAST(user_id AS INTEGER) AS user_id, "
                    "CAST(amount_usd AS DOUBLE) AS amount_usd, "
                    "status "
                    "FROM raw_transactions "
                    "WHERE status = 'completed'"
                ),
                upstream_dependencies=[],
            )
        )

        env.step(
            DatadagAction(
                pipeline_command="create_node",
                model_identifier="mart_user_lifetime_revenue",
                sql_syntax=(
                    "SELECT u.user_id, u.country, "
                    "COALESCE(SUM(t.amount_usd), 0.0) AS lifetime_revenue "
                    "FROM raw_users u "
                    "LEFT JOIN stg_transactions t ON u.user_id = t.user_id "
                    "GROUP BY u.user_id, u.country"
                ),
                upstream_dependencies=["stg_transactions"],
            )
        )

        final_step = env.step(DatadagAction(pipeline_command="execute_dag"))

        return {
            "score": env.state.final_score,
            "done": final_step.done,
            "reward": final_step.reward,
            "trace": final_step.execution_trace,
            "rubric": final_step.metadata.get("task1_rubric"),
        }

    def _create_node(self, action: DatadagAction) -> str:
        model_identifier = action.model_identifier or ""
        self._validate_identifier(model_identifier)

        if model_identifier in self._nodes:
            raise ValueError(f"Node '{model_identifier}' already exists")

        for dep in action.upstream_dependencies:
            if dep not in self._nodes:
                raise ValueError(f"Unknown dependency '{dep}'")

        self._nodes[model_identifier] = NodeSpec(
            sql=action.sql_syntax or "",
            deps=list(action.upstream_dependencies),
            version=1,
        )
        self._assert_acyclic()
        self._validate_compilation_for_current_graph(model_identifier)
        return f"Created node '{model_identifier}' with {len(action.upstream_dependencies)} dependencies."

    def _update_node(self, action: DatadagAction) -> str:
        model_identifier = action.model_identifier or ""
        self._validate_identifier(model_identifier)

        if model_identifier not in self._nodes:
            raise ValueError(f"Node '{model_identifier}' does not exist")

        for dep in action.upstream_dependencies:
            if dep not in self._nodes:
                raise ValueError(f"Unknown dependency '{dep}'")

        node = self._nodes[model_identifier]
        node.sql = action.sql_syntax or node.sql
        node.deps = list(action.upstream_dependencies)
        node.version += 1
        self._assert_acyclic()
        self._validate_compilation_for_current_graph(model_identifier)
        return f"Updated node '{model_identifier}' to version {node.version}."

    def _view_lineage(self, model_identifier: Optional[str]) -> str:
        if not self._nodes:
            return "No nodes in repository."

        if model_identifier:
            if model_identifier not in self._nodes:
                raise ValueError(f"Node '{model_identifier}' does not exist")
            node = self._nodes[model_identifier]
            return (
                f"{model_identifier}: deps={node.deps}, version={node.version}, "
                f"sql_preview={node.sql[:120]}"
            )

        ordered = self._topological_order()
        lines = ["Lineage order: " + " -> ".join(ordered)]
        for node_name in ordered:
            node = self._nodes[node_name]
            lines.append(f"- {node_name}: deps={node.deps}, version={node.version}")
        return "\n".join(lines)

    def _execute_dag(self):
        if not self._nodes:
            raise ValueError("No nodes defined. Create at least one node before execute_dag.")

        order = self._topological_order()
        con = self._seed_duckdb()

        execution_lines: List[str] = ["Executing DAG..."]
        for idx, node_name in enumerate(order):
            node = self._nodes[node_name]
            try:
                con.execute(f"CREATE OR REPLACE TABLE {node_name} AS {node.sql}")
                execution_lines.append(f"[{idx+1}/{len(order)}] OK: {node_name}")
            except Exception as e:
                downstream = self._downstream_of(node_name)
                execution_lines.append(f"[{idx+1}/{len(order)}] ERROR: {node_name}: {e}")
                self._last_error_step = self._state.step_count
                self._state.last_error = str(e)
                self._last_error_signature = str(e)
                self._last_integrity = "compilation_error"
                return "\n".join(execution_lines), False, None, downstream, 0.0, None

        data_sample = None
        if "mart_user_lifetime_revenue" in self._nodes:
            rows = con.execute(
                "SELECT * FROM mart_user_lifetime_revenue ORDER BY user_id LIMIT 5"
            ).fetchall()
            data_sample = [list(r) for r in rows]

        score, rubric = self._grade_task1(con)
        execution_lines.append(f"Task score: {score:.3f}")
        if rubric is not None:
            execution_lines.append(
                "Rubric: "
                f"column={rubric['subscores']['column_score']:.3f}, "
                f"value={rubric['subscores']['value_score']:.3f}, "
                f"grain={rubric['subscores']['grain_score']:.3f}"
            )
        return "\n".join(execution_lines), True, data_sample, [], score, rubric

    def _validate_compilation_for_current_graph(self, focus_node: str) -> None:
        """Compile graph in an isolated DuckDB connection to surface SQL issues early."""
        con = self._seed_duckdb()
        order = self._topological_order()
        for node_name in order:
            node = self._nodes[node_name]
            try:
                con.execute(f"CREATE OR REPLACE TABLE {node_name} AS {node.sql}")
            except Exception as e:
                if node_name == focus_node:
                    raise ValueError(f"SQL compilation failed for '{node_name}': {e}")
                raise ValueError(
                    f"Dependency compilation failed at '{node_name}' while validating '{focus_node}': {e}"
                )

    def _grade_task1(self, con: duckdb.DuckDBPyConnection):
        if "mart_user_lifetime_revenue" not in self._nodes:
            return 0.0, {
                "weights": {
                    "column_score": 0.25,
                    "value_score": 0.50,
                    "grain_score": 0.25,
                },
                "subscores": {
                    "column_score": 0.0,
                    "value_score": 0.0,
                    "grain_score": 0.0,
                },
                "details": {
                    "reason": "Target model mart_user_lifetime_revenue is missing",
                },
                "score": 0.0,
            }

        weights = {
            "column_score": 0.25,
            "value_score": 0.50,
            "grain_score": 0.25,
        }

        expected_columns = ["user_id", "country", "lifetime_revenue"]
        expected_values = {
            1: 150.50,
            2: 20.00,
            3: 0.00,
            4: 75.25,
        }

        try:
            actual_columns = [
                row[0]
                for row in con.execute(
                    "DESCRIBE mart_user_lifetime_revenue"
                ).fetchall()
            ]
        except Exception as e:
            return 0.0, {
                "weights": weights,
                "subscores": {
                    "column_score": 0.0,
                    "value_score": 0.0,
                    "grain_score": 0.0,
                },
                "details": {
                    "reason": f"Could not inspect target schema: {e}",
                },
                "score": 0.0,
            }

        required_present = [c for c in expected_columns if c in actual_columns]
        column_score = len(required_present) / len(expected_columns)

        if "user_id" not in actual_columns:
            grain_score = 0.0
            value_score = 0.0
            details = {
                "expected_columns": expected_columns,
                "actual_columns": actual_columns,
                "missing_columns": [c for c in expected_columns if c not in actual_columns],
                "value_matches": 0,
                "value_expected_count": len(expected_values),
                "grain": {
                    "row_count": 0,
                    "null_user_id_count": None,
                    "duplicate_user_ids": None,
                    "expected_user_count": len(expected_values),
                },
            }
            final_score = (
                weights["column_score"] * column_score
                + weights["value_score"] * value_score
                + weights["grain_score"] * grain_score
            )
            final_score = max(0.0, min(1.0, final_score))
            rubric = {
                "weights": weights,
                "subscores": {
                    "column_score": column_score,
                    "value_score": value_score,
                    "grain_score": grain_score,
                },
                "details": details,
                "score": final_score,
            }
            return final_score, rubric

        try:
            result = con.execute(
                "SELECT user_id, "
                "CASE WHEN lifetime_revenue IS NULL THEN NULL ELSE ROUND(lifetime_revenue, 2) END "
                "AS lifetime_revenue "
                "FROM mart_user_lifetime_revenue"
            ).fetchall()
        except Exception:
            result = []

        user_ids = [row[0] for row in result]
        non_null_user_ids = [uid for uid in user_ids if uid is not None]
        unique_user_ids = set(non_null_user_ids)

        duplicate_count = max(0, len(non_null_user_ids) - len(unique_user_ids))
        null_count = len(user_ids) - len(non_null_user_ids)
        expected_user_set = set(expected_values.keys())
        has_exact_user_set = unique_user_ids == expected_user_set
        has_no_dupes = duplicate_count == 0
        has_no_null = null_count == 0

        grain_checks = [has_exact_user_set, has_no_dupes, has_no_null]
        grain_score = sum(1 for ok in grain_checks if ok) / len(grain_checks)

        got: Dict[int, float] = {}
        for user_id, value in result:
            if user_id is None or value is None:
                continue
            try:
                got[int(user_id)] = float(value)
            except (TypeError, ValueError):
                continue

        value_matches = 0
        for user_id, expected_value in expected_values.items():
            actual_value = got.get(user_id)
            if actual_value is not None and abs(actual_value - expected_value) < 0.01:
                value_matches += 1
        value_score = value_matches / len(expected_values)

        final_score = (
            weights["column_score"] * column_score
            + weights["value_score"] * value_score
            + weights["grain_score"] * grain_score
        )
        final_score = max(0.0, min(1.0, final_score))

        rubric = {
            "weights": weights,
            "subscores": {
                "column_score": column_score,
                "value_score": value_score,
                "grain_score": grain_score,
            },
            "details": {
                "expected_columns": expected_columns,
                "actual_columns": actual_columns,
                "missing_columns": [c for c in expected_columns if c not in actual_columns],
                "value_matches": value_matches,
                "value_expected_count": len(expected_values),
                "grain": {
                    "row_count": len(result),
                    "null_user_id_count": null_count,
                    "duplicate_user_ids": duplicate_count,
                    "expected_user_count": len(expected_values),
                    "has_exact_user_set": has_exact_user_set,
                },
            },
            "score": final_score,
        }

        return final_score, rubric

    def _seed_duckdb(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(database=":memory:")

        con.execute(
            """
            CREATE TABLE raw_users (
                user_id INTEGER,
                country VARCHAR,
                signup_date DATE
            )
            """
        )
        con.execute(
            """
            INSERT INTO raw_users VALUES
                (1, 'US', '2023-01-01'),
                (2, 'IN', '2023-02-10'),
                (3, 'US', '2023-03-15'),
                (4, 'BR', '2023-04-20')
            """
        )

        con.execute(
            """
            CREATE TABLE raw_transactions (
                transaction_id INTEGER,
                user_id VARCHAR,
                amount_usd VARCHAR,
                status VARCHAR
            )
            """
        )
        con.execute(
            """
            INSERT INTO raw_transactions VALUES
                (1001, '1', '100.50', 'completed'),
                (1002, '1', '50.00', 'completed'),
                (1003, '2', '20.00', 'completed'),
                (1004, '2', '80.00', 'refunded'),
                (1005, '4', '75.25', 'completed')
            """
        )

        return con

    def _validate_identifier(self, identifier: str) -> None:
        if not _IDENTIFIER_RE.match(identifier):
            raise ValueError(f"Invalid model_identifier '{identifier}'")

    def _assert_acyclic(self) -> None:
        try:
            self._topological_order()
        except ValueError:
            self._last_integrity = "circular_dependency"
            raise

    def _topological_order(self) -> List[str]:
        indegree: Dict[str, int] = {k: 0 for k in self._nodes}
        reverse: Dict[str, List[str]] = {k: [] for k in self._nodes}

        for node_name, node in self._nodes.items():
            for dep in node.deps:
                if dep not in self._nodes:
                    raise ValueError(f"Unknown dependency '{dep}' for node '{node_name}'")
                indegree[node_name] += 1
                reverse[dep].append(node_name)

        queue = [k for k, v in indegree.items() if v == 0]
        order: List[str] = []

        while queue:
            current = queue.pop(0)
            order.append(current)
            for child in reverse[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(self._nodes):
            raise ValueError("Circular dependency detected")

        return order

    def _downstream_of(self, node_name: str) -> List[str]:
        children: Dict[str, List[str]] = {k: [] for k in self._nodes}
        for n, spec in self._nodes.items():
            for dep in spec.deps:
                children[dep].append(n)

        out: List[str] = []
        stack = list(children.get(node_name, []))
        seen = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            out.append(current)
            stack.extend(children.get(current, []))

        return sorted(out)
