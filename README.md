---
title: SchemaOpt OpenEnv
emoji: 🧠
colorFrom: blue
colorTo: green
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

## What this environment is

SchemaOpt is a multi-step optimization environment for analytical data workloads.

In each episode, the agent:

- inspects clustered workload structure
- creates, modifies, or drops derived objects such as aggregate materializations and denormalized tables
- benchmarks routed rewrites against the baseline workload
- submits a final solution under storage, refresh, and step budgets

The environment is designed to model a realistic warehouse-tuning loop rather than a single SQL-generation task. The agent does not rewrite workload SQL directly. Instead, it changes the physical schema available to the query router and is judged on correctness-gated performance gains.

## Quick Start

### 1. Run locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
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

## How to run it

### Local server

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Local inference runner

```bash
python inference.py --task-id schemaopt_easy_hiring_pipeline --model-name gpt-5.4-mini
```

### Docker

```bash
docker build -t schemaopt-openenv .
docker run --rm -p 8000:8000 schemaopt-openenv
```

## How to reset and step it

The environment follows the standard OpenEnv HTTP contract.

### Python example

Reset starts a new episode and optionally selects a task. Step sends one action and returns the next observation, reward, and done flag.

```python
from client import SchemaOptEnv
from models import SchemaOptAction

async def main():
    async with SchemaOptEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_id="schemaopt_easy_hiring_pipeline")
        print(result.observation.decision_state)

        result = await env.step(
            SchemaOptAction(
                operation="get_cluster_context",
                cluster_id="schemaopt_easy_hiring_pipeline_cluster_03",
            )
        )
        print(result.observation.action_feedback)
```

For synchronous usage:

```python
from client import SchemaOptEnv
from models import SchemaOptAction

with SchemaOptEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="schemaopt_easy_hiring_pipeline")
    result = env.step(
        SchemaOptAction(
            operation="get_cluster_context",
            cluster_id="schemaopt_easy_hiring_pipeline_cluster_03",
        )
    )
    print(result.observation.decision_state)
```

Typical optimization loop:

1. `reset(task_id=...)`
2. `get_cluster_context`
3. `create_derived_object` or `modify_derived_object`
4. `inspect_rewrite_status` or `benchmark_cluster`
5. repeat until `submit` or budget exhaustion

Episodes also auto-submit when the task step budget is exhausted.

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

## Action, Observation, and State Summary

### Actions

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

### Observations

Observations include:

- catalog summary
- workload summary
- benchmark context
- router summary
- action feedback
- decision state

### State

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

Task assets live in task_assets and task_assets/databases.

SchemaOpt tasks are grouped by difficulty and by workload family. Easy tasks usually reward a small number of obvious reusable rollups, medium tasks require broader reuse across related clusters, and hard tasks demand cross-cluster tradeoffs under tighter budget pressure.

### Easy Tasks

These tasks emphasize compact aggregate reuse, clearer hotspot structure, and lower exploration cost.

| Task ID | Workload | What it tests |
|---|---|---|
| `schemaopt_easy_geo_metrics` | Geographic metrics benchmark | Straightforward country, city, and regional reporting patterns with compact reusable group-by objects. |
| `schemaopt_easy_hiring_pipeline` | Recruiting operations | Funnel, requisition, and posting-health reporting with time-bucketed operational aggregates. |
| `schemaopt_easy_product_adoption` | Product adoption analytics | Workspace usage, onboarding, and feature-adoption rollups with relatively direct reuse paths. |
| `schemaopt_easy_retail_ops` | Retail operations benchmark | Small exact-match and filtered aggregate opportunities over transactional retail-style queries. |

### Medium Tasks

These tasks require broader cluster coverage, more mixed grouping patterns, and better object reuse decisions.

| Task ID | Workload | What it tests |
|---|---|---|
| `schemaopt_medium_campaign_performance` | Paid acquisition analytics | Reusable reporting across portfolio, campaign, ad-group, and keyword slices. |
| `schemaopt_medium_customer_ops` | Customer operations benchmark | More heterogeneous aggregate and join-based reuse over customer-service style analytics. |
| `schemaopt_medium_delivery_operations` | Delivery operations analytics | Portfolio, execution, backlog, and collaboration reporting across multi-dimensional delivery views. |
| `schemaopt_medium_motorsport_ops` | Motorsport operations benchmark | Clustered analytical queries over racing and event data with broader aggregate reuse choices. |

### Hard Tasks

These tasks put more pressure on global planning: more clusters, more competing materialization choices, and greater risk of building narrow objects that do not generalize.

| Task ID | Workload | What it tests |
|---|---|---|
| `schemaopt_hard_lifecycle_engagement` | Lifecycle marketing and deliverability | Campaign influence, template health, engagement lift, churn signals, and deliverability risk across many clustered reporting families. |
| `schemaopt_hard_mobile_revenue_ops` | Mobile revenue operations | Install, release, geography, platform, and monetization reporting with stronger cross-cluster tradeoffs. |
| `schemaopt_hard_sports_analytics` | Sports analytics benchmark | More varied exact-match and grouped aggregate opportunities across a larger benchmark-derived analytical workload. |

## How to deploy / push

### Hugging Face Spaces / OpenEnv push

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
|- client.py
|- inference.py
|- models.py
|- openenv.yaml
|- pyproject.toml
|- requirements.txt
|- tasks.py
|- server/
|  |- app.py
|  |- rubrics.py
|  |- schemaopt_environment.py
|- scripts/
|- task_assets/
|  |- *.json
|  |- databases/*.duckdb
```

## Notes

- The environment can auto-submit when the step budget is exhausted.
- Large or low-utility derived objects increase pressure and reduce rewards.


