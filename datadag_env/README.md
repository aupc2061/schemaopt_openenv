# DataDAG Environment

DataDAG is an OpenEnv-compatible reinforcement learning environment for repository-level SQL DAG orchestration.

## v1 scope

- Task 1 (easy): build a two-stage pipeline from raw tables.
- Commands: `create_node`, `update_node`, `view_lineage`, `execute_dag`.
- Backend: DuckDB.
- Grader: deterministic score in [0.0, 1.0].

## Action

`DatadagAction` fields:

- `pipeline_command`: one of `create_node | update_node | execute_dag | view_lineage`
- `model_identifier`: model name for create/update/view
- `sql_syntax`: SQL text for create/update
- `upstream_dependencies`: parent models for DAG ordering

## Observation

`DatadagObservation` fields:

- `dag_integrity_status`: `valid | compilation_error | circular_dependency`
- `execution_trace`: compiler/execution feedback
- `data_sample`: optional 5-row sample from target table
- `cascading_failure_nodes`: downstream nodes affected by an error
- inherited `reward`, `done`, `metadata`

## Run locally

```bash
cd datadag_env
uv run --project . server
```

Or:

```bash
cd datadag_env
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /tasks`
- `GET /grader`
- `POST /baseline`
