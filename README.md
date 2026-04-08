---
title: SchemaOpt OpenEnv
emoji: 🧠
colorFrom: blue
colorTo: teal
sdk: docker
pinned: false
app_port: 8000
base_path: /web
suggested_hardware: cpu-basic
tags:
  - openenv
  - fastapi
  - duckdb
  - benchmark
---

# SchemaOpt OpenEnv

SchemaOpt is an OpenEnv environment for workload-adaptive warehouse optimization. Agents interact with real DuckDB workloads, propose derived objects, and are scored on correctness-gated performance improvement under realistic budgets.

## Quick Start

### 1. Run locally

```bash
pip install -r requirements.txt
uvicorn schemaopt_env.server.app:app --host 0.0.0.0 --port 8000
```

### 2. Smoke test

```bash
curl http://localhost:8000/tasks
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
```

### 3. Run inference

```bash
python inference.py --tasks schemaopt_easy_hiring_pipeline,schemaopt_medium_campaign_performance,schemaopt_hard_mobile_revenue_ops
```

## Why this environment

Schema optimization is a practical data-platform problem: choosing joins, aggregates, and denormalized structures that improve repeated analytics while respecting storage and refresh costs. SchemaOpt benchmarks this process with explicit actions, deterministic scoring, and holdout generalization.

## Environment Highlights

- Real DuckDB execution on isolated per-episode database copies
- Typed action, observation, and state models
- Correctness-gated query rewrites and routed benchmarking
- Dense step rewards plus final unified grading
- 11 tasks across easy, medium, and hard difficulty

## API Surface

Endpoints:

- POST /reset
- POST /step
- GET /state
- GET /schema
- GET /tasks
- GET /grader
- POST /baseline

## Action Space

Supported operations:

- `inspect_catalog`
- `inspect_table_stats`
- `get_cluster_context`
- `inspect_rewrite_status`
- `create_derived_object`
- `modify_derived_object`
- `drop_derived_object`
- `benchmark_subset`
- `benchmark_cluster`
- `submit`

Key validation constraints:

- inspect_table_stats requires target_id
- get_cluster_context requires cluster_id
- inspect_rewrite_status requires exactly one scope: target_id, cluster_id, or query_ids
- create_derived_object and modify_derived_object require object_kind, name, sql_definition, source_objects
- drop_derived_object requires target_id
- benchmark_subset requires query_ids
- benchmark_cluster requires cluster_id

## Observation and State

Observations include:

- catalog summary
- workload summary
- benchmark context
- router summary
- action feedback
- decision state

State tracks:

- step count and difficulty
- remaining step and object budget
- remaining storage and refresh budget
- current focus cluster
- cluster attempt and benchmark history
- useful vs unused derived objects

## Scoring

Step-time rewards are rubric-based:

- Error penalties for validation and runtime failures
- Utility/pressure-based rewards for create and modify
- Cleanup incentives for dropping low-value objects
- Delta-style rewards for benchmark actions
- Submit reward equals final score

Final score combines:

- 0.45 \* visible_gated_improvement
- 0.20 \* holdout_gated_improvement
- 0.20 \* correctness
- 0.10 \* migration_score
- 0.05 \* storage_score

## Tasks

Task assets live in schemaopt_env/task_assets and schemaopt_env/task_assets/databases.

Available task IDs:

- schemaopt_easy_geo_metrics
- schemaopt_easy_hiring_pipeline
- schemaopt_easy_product_adoption
- schemaopt_easy_retail_ops
- schemaopt_medium_campaign_performance
- schemaopt_medium_customer_ops
- schemaopt_medium_delivery_operations
- schemaopt_medium_motorsport_ops
- schemaopt_hard_lifecycle_engagement
- schemaopt_hard_mobile_revenue_ops
- schemaopt_hard_sports_analytics

## Deployment

### Docker

```bash
docker build -t schemaopt-openenv .
docker run --rm -p 8000:8000 schemaopt-openenv
```

### Hugging Face Spaces

```bash
openenv push
```

Optional:

```bash
openenv push --namespace my-org --private
openenv push --repo-id my-org/schemaopt-openenv
```

Deployed routes typically include:

- /web
- /docs
- /health
- /ws

## Inference Runner

The submission runner is inference.py at the repository root.

Common usage:

```bash
python inference.py --task-id schemaopt_easy_hiring_pipeline --model-name gpt-5.4-mini --max-steps 40 --max-action-retries 4
```

Environment variables:

- OPENAI_API_KEY
- HF_TOKEN
- API_BASE_URL
- MODEL_NAME
- MAX_STEPS
- TASK_ID
- MAX_ACTION_RETRIES
- HF_SPACE_REPO_ID or SPACE_ID (for runtime task asset materialization when using Git LFS pointers)

## Project Layout

```text
metaenv/
|- Dockerfile
|- inference.py
|- openenv.yaml
|- requirements.txt
|- schemaopt_env/
|  |- client.py
|  |- models.py
|  |- tasks.py
|  |- openenv.yaml
|  |- server/
|  |  |- app.py
|  |  |- rubrics.py
|  |  |- schemaopt_environment.py
|  |- task_assets/
|     |- *.json
|     |- databases/*.duckdb
```

## Notes

- The environment can auto-submit when the step budget is exhausted.
- Large or low-utility derived objects increase pressure and reduce rewards.
