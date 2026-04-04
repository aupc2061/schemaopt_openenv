# SchemaOpt Environment

SchemaOpt is a standalone OpenEnv benchmark for workload-adaptive warehouse optimization.

## Current Fidelity Model

- Spider databases and text-to-SQL pairs are the primary workload source.
- Each episode is anchored to a real Spider `db_id` and its SQLite database file.
- Episode reset attaches the Spider database in DuckDB and creates only the `derived` schema dynamically.
- Query benchmarking uses real DuckDB execution plus `EXPLAIN`-derived structural metrics.
- Query rewrites are explicit: routed execution must correspond to a concrete rewritten SQL query against a derived object.
- Correctness is gated by actual result equivalence against the original query output.

## Scope

- Fixed analytical query workloads from Spider text-to-SQL pairs
- Deterministic query retrieval over Spider workload metadata
- Logical derived objects: join materializations, aggregate materializations, filtered projections, denormalized tables
- Explicit rewritten-SQL routing to derived objects
- Correctness-gated performance reward

## Module Layout

- Shared workload dataclasses live in `models.py`
- Spider registry loading and sampling live in `spider_registry.py`
- Runtime episode logic lives in `server/schemaopt_environment.py`
- The debug plan tracer remains under `scripts/debug_walk_plan.py`

## Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /tasks`
- `GET /grader`
- `POST /baseline`
