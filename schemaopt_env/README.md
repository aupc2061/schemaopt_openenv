# SchemaOpt Environment

SchemaOpt is a standalone OpenEnv benchmark for workload-adaptive warehouse optimization.

## Current Fidelity Model

- Task definitions are manifest-backed and resolved from local benchmark source assets.
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

## Inference Runner Notes

The standalone runner in `../schemaopt_inference.py` uses a strict action validation path.

- Normalization is intentionally minimal.
- `inspect_rewrite_status` supports `query_id` as a compatibility alias and maps it to `query_ids`.
- Other malformed scope payloads are not silently rewritten; they are rejected and retried so model-output bugs stay visible.

Result files include a `run_summary` section with:

- configured/effective step settings
- retry settings
- termination reason
- warnings
- recent action trace tail

## UV Commands

From the repository root:

```powershell
uv pip install --python e:/Project_Repos/metaenv/.venv/Scripts/python.exe pytest
e:/Project_Repos/metaenv/.venv/Scripts/python.exe -m pytest .\tests\test_schemaopt_inference.py -q
```

Quick sanity inference (1 step):

```powershell
e:/Project_Repos/metaenv/.venv/Scripts/python.exe .\schemaopt_inference.py --model-name gpt-5.4-mini --task-id schemaopt_easy_hiring_pipeline --max-steps 1 --max-action-retries 2 --output .\inference_sanity.json
```
