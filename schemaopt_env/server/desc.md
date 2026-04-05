# `schemaopt_env` Summary

## Overview
`schemaopt_env` is a standalone OpenEnv benchmark for workload-aware SQL schema optimization on DuckDB.  
Instead of editing pipeline SQL files, the agent interacts with a live database-backed environment and tries to improve a fixed analytical workload by creating derived objects such as materialized aggregates or denormalized helper tables.

Each episode is defined by:
- a real DuckDB database copied into an episode workspace
- a visible query workload
- a hidden holdout workload
- precomputed query clusters and metadata
- budgets on number of derived objects, storage, refresh/runtime cost, and step count

The core objective is:

> reduce measured query execution cost while preserving exact query results

---

## Problem Formulation
The environment is formulated as a multi-step RL-style optimization problem.

At a high level:
- **State**: the agent observes database/catalog summaries, workload cluster summaries, retrieved query context, benchmark results, and feedback from previous actions
- **Action**: the agent can inspect workload/query metadata, retrieve subsets of queries, create/modify/drop derived objects, benchmark subsets/clusters, and submit a final solution
- **Transition**: derived objects are materialized in DuckDB, routing is recomputed, benchmark metrics are updated, and the observation changes accordingly
- **Reward**: the agent receives dense step rewards for benchmarking outcomes and a terminal score at submission

---

## State
The observation returned by the environment is structured around:

### 1. Task Context
- task id
- difficulty
- domain
- objective
- budgets
- allowed object kinds

### 2. Catalog Summary
- source schemas/tables
- column information
- derived objects created so far
- storage usage estimate
- refresh/runtime usage estimate

### 3. Workload Summary
- number of visible queries
- hotspot clusters
- cluster summaries
- preferred object kind per cluster
- cluster-level tables, columns, plan features, and weights

### 4. Retrieval Context
- last retrieval request
- matched query ids
- matched cluster ids
- retrieval count

### 5. Benchmark Context
- baseline weighted cost
- current weighted cost
- raw improvement
- gated improvement
- correctness coverage
- routed query count
- incorrect query count
- latest plan deltas

### 6. Action Feedback
- validation errors
- query summaries/context
- plan summaries
- derived object summaries
- benchmark summaries
- final submission summary

---

## Action Space
The action space is explicit and database-oriented.

### Inspection / Retrieval
- `inspect_catalog`
- `inspect_table_stats`
- `inspect_cluster`
- `inspect_query`
- `inspect_query_plan`
- `inspect_router_status`
- `retrieve_queries`
- `get_query_context`
- `list_derived_objects`

### Optimization
- `create_derived_object`
- `modify_derived_object`
- `drop_derived_object`
- `checkpoint`
- `revert_checkpoint`

### Evaluation
- `benchmark_subset`
- `benchmark_cluster`
- `submit`

For `create_derived_object` / `modify_derived_object`, the agent provides:
- `object_kind`
- `name`
- `sql_definition`
- `source_objects`
- `grain_hint`
- `intended_clusters`
- `routing_tags`

---

## Rewards
The environment uses both dense step rewards and a terminal score.

### Dense Rewards
The agent can gain or lose reward based on:
- successful benchmarking
- measured workload improvement
- correctness preservation
- budget penalties
- small shaping around object creation / evaluation flow

### Terminal Reward / Final Score
At `submit`, the environment computes a final score from:
- visible workload performance
- holdout workload performance
- correctness coverage
- migration / unused-object discipline
- storage and refresh efficiency

Important property:
- performance improvement is correctness-gated
- faster but incorrect rewrites do not get rewarded as valid optimization

---

## Core Functionalities
The current environment supports:

### Real DuckDB-backed execution
- task-specific DuckDB databases are loaded from packaged task assets
- derived objects are materialized in a `derived` schema
- query benchmarking runs against actual DuckDB execution

### Query-plan inspection
- plan summaries are extracted from DuckDB `EXPLAIN`
- plan depth, operator count, join count, and blocking operators are tracked

### Deterministic workload retrieval
- the agent does not see the whole workload at reset
- it can retrieve queries by cluster or other filters during the episode

### Rewriting / Routing
- workload queries remain fixed
- the environment tries to route a query through a compatible derived object
- routed queries are validated against original-query results

### Benchmarking
- the agent can benchmark subsets or clusters before final submission
- visible and holdout workloads are scored separately

---

## Current Task Suite
The current suite contains **6 curated tasks** designed as a compact benchmark set for workload-aware schema optimization.

### Easy Tasks
Easy tasks have:
- 12 visible queries
- 6 holdout queries
- 3 clusters
- max 3 new derived objects
- max 18 steps

Tasks:
- `schemaopt_easy_hiring_pipeline`
- `schemaopt_easy_product_adoption`

Task description:
- optimize a small workload over one business domain
- clusters are relatively compact and usually centered around a small number of base tables
- intended to test whether an agent can identify obvious aggregation/materialization opportunities

---

### Medium Tasks
Medium tasks have:
- 24 visible queries
- 12 holdout queries
- 4 clusters
- max 4 new derived objects
- max 24 steps

Tasks:
- `schemaopt_medium_campaign_performance`
- `schemaopt_medium_delivery_operations`

Task description:
- optimize a broader workload with more clusters and more query families
- tests whether the agent can choose derived objects that help multiple related queries instead of one-off rewrites
- introduces stronger tradeoffs around object budgets and cluster prioritization

---

### Hard Tasks
Hard tasks have:
- 35 visible queries
- 20 holdout queries
- 5 clusters
- max 5 new derived objects
- max 30 steps

Tasks:
- `schemaopt_hard_mobile_revenue_ops`
- `schemaopt_hard_lifecycle_engagement`

Task description:
- optimize larger workloads with multiple hotspot clusters and more cross-query variation
- intended to test multi-step planning, query retrieval strategy, object selection under budget, and final submission discipline
- holdout workloads test whether created objects generalize beyond only the visible queries

---

## Current Task Objective Template
All current tasks use the same objective pattern:

> Optimize the `<domain>` workload over the real DuckDB source database by materializing derived objects that reduce measured execution cost while preserving exact query results.

---

## Practical Characterization Of The Current Environment
In its current form, `schemaopt_env` is best described as:
- a **database-backed RL-style benchmark**
- for **workload-driven SQL schema optimization**
- with **retrieval-based query context**
- **real DuckDB execution and benchmarking**
- and **submission-time evaluation on visible + holdout workloads**

If you want, I can turn this into a `SCHEMAOPT_ENV_SUMMARY.md` file in the repo.
