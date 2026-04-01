# SchemaOpt Environment

SchemaOpt is a standalone OpenEnv benchmark for workload-adaptive warehouse optimization.

## Current Fidelity Model

- Task definitions are manifest-backed and resolved from real `dacomp-de-impl` dataset assets.
- Each task points to a real DuckDB database file and a precomputed workload manifest.
- Episode reset copies the task database into an isolated workspace and creates only the `derived` schema dynamically.
- Query benchmarking uses real DuckDB execution plus `EXPLAIN`-derived structural metrics.
- Query rewrites are explicit: routed execution must correspond to a concrete rewritten SQL query against a derived object.
- Correctness is gated by actual result equivalence against the original query output.

## Scope

- Fixed analytical query workloads
- Clustered workload summaries plus deterministic query retrieval over precomputed metadata
- Logical derived objects: join materializations, aggregate materializations, filtered projections, denormalized tables
- Explicit rewritten-SQL routing to derived objects
- Correctness-gated performance reward

## Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /tasks`
- `GET /grader`
- `POST /baseline`
