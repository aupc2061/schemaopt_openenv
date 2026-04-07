# Context Summary

## Project Goal
The repo now contains a standalone OpenEnv benchmark named `schemaopt_env` for workload-aware schema optimization on DuckDB.

The intended problem setup is:
- fixed analytical workload SQL
- real DuckDB-backed task database
- agent can create/modify/drop derived objects
- environment evaluates whether derived objects reduce workload cost while preserving exact query semantics
- visible and holdout workloads are both scored

## Main Environment Shape
`schemaopt_env` is implemented as a separate package, not inside `datadag_env`.

Core files:
- `schemaopt_env/models.py`
- `schemaopt_env/tasks.py`
- `schemaopt_env/server/app.py`
- `schemaopt_env/server/schemaopt_environment.py`
- `schemaopt_inference.py`

Main API / interaction surface:
- `reset`
- `step`
- `state`
- `schema`
- `tasks`
- `grader`
- `baseline`

Observation/state includes:
- task metadata and budgets
- catalog summary
- workload cluster summary
- retrieval context
- benchmark context
- action feedback
- router summary

Action space includes:
- inspection actions such as `inspect_cluster`, `inspect_query`, `inspect_query_plan`, `inspect_router_status`
- retrieval actions such as `retrieve_queries`, `get_query_context`
- optimization actions such as `create_derived_object`, `modify_derived_object`, `drop_derived_object`
- evaluation actions such as `benchmark_cluster`, `benchmark_subset`, `submit`

## Task Packaging Direction
The original synthetic task generation was replaced with manifest-backed task assets under:
- `schemaopt_env/task_assets`

Current task suite:
- 6 curated benchmark tasks
- 2 easy
- 2 medium
- 2 hard

Example tasks:
- `schemaopt_easy_hiring_pipeline`
- `schemaopt_hard_mobile_revenue_ops`

Seed provenance uses local benchmark source assets.

## Major Fidelity Refactor Completed
We refactored the env away from the earlier synthetic benchmark behavior toward a more DuckDB-grounded setup.

Major changes already made:
- real task manifests instead of runtime synthetic task generation
- real DuckDB database path per task
- richer SQL/query metadata in task assets
- plan extraction and execution-based benchmarking path in env
- result-equivalence based correctness checks
- deterministic retrieval over task metadata
- top-level router diagnostics

## Important Environment Fixes Already Made

### 1. Removed brittle routing based on raw predicate strings
Routing eligibility was made more semantic by canonicalizing:
- source tables
- predicates
- group-by shape

This was done so alias-only differences such as `b."updated_at"` vs `updated_at` do not block rewrite matching.

### 2. Added richer router diagnostics
The environment now surfaces:
- dominant rejection reasons
- candidate object coverage
- top rewrite candidate
- object usefulness / emptiness diagnostics

### 3. Added duplicate / empty object handling
Derived objects now have canonical signatures.

Behavior added:
- duplicate-signature objects are penalized / not rewarded
- empty objects are penalized
- create reward is no longer unconditional

### 4. Improved inference prompt and control loop
`schemaopt_inference.py` was updated so the model is guided to:
- copy predicates from query context closely
- inspect router status or benchmark after create/modify
- avoid repeated duplicate creations
- submit when repeated evidence shows no improvement

### 5. Fixed a critical parser bug in `schemaopt_environment.py`
One major env bug was in `_parse_sql_metadata(...)`.

The regexes for:
- `WHERE`
- `GROUP BY`

were corrupted and effectively not matching, so exact derived SQL could parse with empty predicates and empty group-by fields. This blocked routing even for exact-match aggregates.

That bug was fixed by restoring proper regexes:
- `r"\\bwhere\\b"`
- `r"\\bgroup\\s+by\\b"`

This fix was the turning point that allowed real routing on easy tasks.

## Inference / Policy Evolution

### Earlier state
Initial inference runs showed:
- agent could act and produce parsable actions
- env could step successfully
- but no real optimization happened
- created objects were often unused
- runs sometimes crashed due to non-JSON-serializable env payloads

We fixed:
- serialization of `PlanArtifact`-like outputs
- name validation guidance in prompts
- model-only action generation without heuristic fallback

### Later easy-task success
After the parser fix and routing improvements, `schemaopt_easy_customer360` produced real optimization.

Observed outcomes:
- non-zero visible improvement
- non-zero holdout improvement
- routed visible queries
- zero incorrect queries
- multiple created objects actually used by workload queries

This established that:
- the env can now support genuine optimization
- the router and benchmark loop are working at least on easy tasks

## Important Hard-Task Diagnosis
The later mobile revenue hard-task run still showed zero improvement.

At first this looked like a model-policy failure, but deeper inspection showed a task-quality issue.

### What was observed
In the hard-task result:
- routed query count stayed `0`
- dominant rejection reason was `empty_derived_object`
- the model’s derived object was created successfully but materialized zero rows

### What was inspected
We inspected:
- `schemaopt_env/task_assets/schemaopt_hard_mobile_revenue_ops.json`
- the local source DuckDB backing the curated mobile revenue task

Cluster 05 representative query was:
```sql
SELECT date_trunc('month', try_cast(b."date" AS TIMESTAMP)) AS metric_period, ...
FROM raw.stats_installs_os_version b
WHERE try_cast(b."date" AS TIMESTAMP) IS NOT NULL AND b."_file" IS NOT NULL
```

### What was discovered
In the actual start DB:
- `raw.stats_installs_os_version.date` is stored as `VARCHAR`
- sample values look like `2020-01-01 00:00:00.000 +0800`
- `try_cast(date AS TIMESTAMP)` returns `NULL` for those strings in this setup

Direct DB checks showed:
- total rows in `raw.stats_installs_os_version`: `1360`
- rows satisfying `try_cast(date AS TIMESTAMP) IS NOT NULL AND _file IS NOT NULL`: `0`
- exact cluster-05 aggregate result rows: `0`

We also checked analogous mobile revenue clusters:
- cluster 04 (`stats_installs_device`): exact query rows `0`
- cluster 03 (`stats_installs_country`): exact query rows `0`

But the data is actually parseable with a format-aware function:
- `try_strptime(date, '%Y-%m-%d %H:%M:%S.%f %z')`

This means the current hard Google Play task asset is flawed for those clusters.

## Current Conclusion

### What is working now
- `schemaopt_env` can run full episodes
- the model can interact with the env without heuristic fallback
- easy tasks can now produce real routing and real measured improvement
- router diagnostics are much better than before

### What is still broken or incomplete
- at least part of `schemaopt_hard_google_play` is task-invalid because representative workload SQL returns empty results on the packaged start DB
- some hard-task failures are therefore task-definition issues, not inference-policy issues
- final score can still be nontrivial even when performance improvement is zero, due to correctness/storage/migration components

## Current Known Action Items

### High priority
- repair the affected Google Play task workload definitions
- replace invalid `try_cast(... AS TIMESTAMP)` usage for timezone-bearing `date` strings with a parse that matches stored data format
- regenerate affected manifests
- validate offline that each cluster’s representative query returns rows on the start DB
- validate offline that each cluster still has a feasible optimization opportunity

### Secondary
- continue improving hard-task policy behavior after task-asset fixes
- strengthen task-generation validation so “reference rewrite feasible” is actually verified against live task DB contents

## Practical Takeaway
As of now:
- the env itself is materially better than the original synthetic version
- the easy benchmark path is functioning as intended
- the current blocker for at least one hard task is bad task SQL / task packaging, not the previously fixed router bug
