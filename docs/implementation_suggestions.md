# SchemaOpt Implementation Suggestions v3

This version incorporates the current `metaenv` baseline plus the team’s parallel reward-densification proposal.

The goal is to improve the benchmark specifically on the parts the hackathon will reward most:

- `Real-world utility (30%)`
- `Task & grader quality (25%)`
- `Environment design (20%)`
- `Creativity & novelty (10%)`

The guiding rule is simple:

- improve incentives, diagnostics, and state realism
- do not add gimmicks
- do not leak hidden answers
- do not make reward shaping more important than the true task objective

## What Is Already Good Enough To Not Be Priority 1

The current baseline has already improved meaningfully in:

- exact rewrite realism
- canonical output reconstruction
- broader aggregate parsing
- route rejection fidelity

So the highest-ROI work is no longer in SQL rewrite mechanics. It is now in:

- episode boundaries
- dense reward shaping
- state design
- scorer / diagnostics legibility
- grounded novelty through SLA and maintenance realism

## Final Recommended Workstreams

## Workstream 1: Fix Episode Boundaries First

### 1. Add proper environment-level termination on `max_steps`

Current gap:

- `max_steps` is present in task budgets
- the environment still effectively terminates only on explicit `submit`

Why this matters:

- directly improves `Environment design`
- prevents brittle dependence on the external inference loop
- gives judges a cleaner notion of episode lifecycle

Implementation:

- In `step()`, after incrementing `step_count`, check `task.budgets["max_steps"]`
- If the budget is exhausted and the agent has not submitted:
  - auto-submit the current design
  - compute the same visible + holdout grading path as `submit`
  - return `done=True`
  - set a clear terminal message
  - mark a state flag like `auto_submitted=True`

Important:

- this should be a forced submit, not a hard failure
- apply at most a mild penalty for not explicitly submitting

## Workstream 2: Redesign Intermediate Reward Shaping

This is the single most important implementation area now.

## Reward principles

- Keep `final_score` unchanged
- Keep `terminal_reward = final_score`
- Densify only step-level shaping
- Do not reward read-only actions
- Make shaping local, interpretable, and hard to farm

### 2. Keep final grading unchanged

Do not change the submission metric:

- `0.45 * visible_gated_improvement`
- `0.20 * holdout_gated_improvement`
- `0.20 * correctness`
- `0.10 * migration`
- `0.05 * storage`

Why:

- current final metric is already aligned with the benchmark
- changing terminal semantics now creates unnecessary evaluation risk

### 3. Replace global benchmark memory with scope-local benchmark memory

Current issue:

- benchmark reward still compares against one global `_latest_visible_benchmark`
- this creates cross-cluster contamination
- a weaker cluster can be unfairly punished because a different cluster benchmarked better earlier

This is a real shaping bug and should be fixed.

Implementation:

- replace global benchmark memory with a map keyed by benchmark scope

Use:

- `cluster:<cluster_id>` for `benchmark_cluster`
- `subset:<stable_hash(sorted_query_ids)>` for `benchmark_subset`

Store for each scope:

- `derived_state_hash`
- `gated_improvement`
- `correctness_coverage`
- `routed_query_ratio`
- `resource_pressure`
- `benchmark_score`

Rules:

- first benchmark for a scope uses no previous scope baseline
- repeated benchmark for the same scope with unchanged derived state returns `0.0`
- only compare a scope against its own prior score

Why this matters:

- directly improves reward fairness
- makes dense reward meaningful and less noisy
- helps the hard tasks feel much more coherent

### 4. Introduce a scoped benchmark score

Recommended approach:

- keep the terminal score untouched
- use a separate scoped benchmark score for intermediate shaping

Recommended benchmark score components:

- `gated_improvement`
- `routed_query_ratio`
- `correctness_coverage`
- `budget_penalty`
- `resource_pressure`

Recommended benchmark reward logic:

- first benchmark for a scope:
  - reward = current scope score
- later benchmark for same scope:
  - reward = current scope score - previous scope score
- unchanged scope + unchanged derived state:
  - reward = `0.0`

Keep benchmark reward bounded:

- suggested clamp: `[-0.20, 0.20]`

Why this is good:

- denser signal
- less cross-scope noise
- encourages iterative improvement inside the same workload slice

### 5. Add soft resource-pressure shaping before hard budget overflow

Current issue:

- hard budget penalties only activate after exceeding limits

This is too late for good policy learning.

Add a soft shaping term:

- `resource_pressure = 0.4 * storage_ratio + 0.4 * refresh_ratio + 0.2 * object_ratio`

Use it only in:

- create / modify / drop shaping
- benchmark shaping

Do not use it to replace hard budget penalties.

Why this is good:

- gives earlier feedback that an object is expensive
- stays realistic
- aligns with warehouse design tradeoffs

### 6. Replace binary create/modify reward with diagnostic-based shaping

Current issue:

- create/modify shaping is still too binary

What to implement:

- compute utility using:
  - `eligible_query_ratio`
  - `eligible_cluster_ratio`

Recommended object utility:

- `object_utility = 0.60 * eligible_query_ratio + 0.40 * eligible_cluster_ratio`

Recommended create shaping:

- duplicate signature: negative
- empty object: stronger negative
- otherwise:
  - small positive reward scaled by `object_utility`
  - subtract mild cost if resource pressure increased

Recommended modify shaping:

- compare new utility against old utility
- reward genuine repair
- penalize degradation

Why this matters:

- gives a meaningful gradient between “helps one query” and “helps many”
- makes `modify_derived_object` strategically useful
- stays grounded in object usefulness rather than arbitrary heuristics

Important:

- keep action-level reward small
- suggested band remains roughly `[-0.04, 0.04]`

### 7. Replace flat drop penalty with usefulness-aware drop shaping

Current issue:

- dropping a bad object is often the right decision
- a flat negative reward discourages cleanup

Recommended behavior:

- dropping empty / duplicate / clearly useless objects:
  - small positive or neutral reward
- dropping useful objects:
  - negative reward
- allow a small bonus if the drop meaningfully reduces resource pressure

Why this is good:

- rewards cleanup when cleanup is the right engineering decision
- prevents object accumulation for the wrong reasons

### 8. Split generic error penalty into typed penalties

Current issue:

- one flat `-0.05` error penalty is too coarse

Recommended split:

- validation / payload errors: mild negative
- SQL runtime errors: moderate negative
- internal env errors: strongest negative
- duplicate-object cases should be handled by the action-specific shaping path, not double-penalized

Also include:

- `error_type` in `action_feedback`

Why this is good:

- keeps the environment strict
- avoids making recoverable mistakes disproportionately expensive

### 9. Keep read-only actions at zero reward

Do not add reward for:

- `inspect_cluster`
- `inspect_query`
- `inspect_query_plan`
- `inspect_router_status`
- `retrieve_queries`
- `get_query_context`
- `list_derived_objects`

Why this should remain true:

- these actions are too easy to farm
- the benchmark should reward optimization progress, not browsing
- judges will view this as cleaner and more honest

If more signal is needed:

- expose progress metadata in observations
- do not pay reward for reading

## Workstream 3: Expand State And Diagnostics

### 10. Expand `SchemaOptState` with realistic operational metrics

Add:

- `steps_remaining`
- `objects_remaining`
- `storage_used_bytes`
- `storage_remaining_bytes`
- `refresh_runtime_used_ms`
- `refresh_runtime_remaining_ms`
- `visible_queries_routed`
- `visible_clusters_helped`
- `unused_object_count`
- `last_benchmark_improvement`
- `best_visible_improvement_so_far`
- `auto_submitted`

Why this matters:

- better environment design
- better policy quality
- better debugging and judge legibility

### 11. Add reward decomposition to action feedback

This is one of the strongest ideas from the team proposal and should be implemented.

For create/modify/drop:

- include `reward_components`
  - `utility_component`
  - `resource_component`
  - `duplicate_penalty`
  - `empty_object_penalty`

For benchmark:

- include `reward_components`
  - `gated_improvement_component`
  - `routing_component`
  - `correctness_component`
  - `budget_component`
  - `resource_component`
  - `scope_previous_score`
  - `scope_current_score`

For errors:

- include `reward_components`
  - `error_penalty`
  - `error_type`

Why this matters:

- improves action-feedback quality
- helps policy debugging
- gives judges and teammates visibility into why reward changed

### 12. Add `inspect_budget_status`

This remains the best new action to add.

Return:

- step usage
- object usage
- storage usage
- refresh usage
- remaining budgets

Why:

- realistic
- useful
- easy to justify

### 13. Add `inspect_object_coverage`

This remains the second best action to add.

Return:

- eligible visible queries
- eligible visible clusters
- rejection reasons for unmatched visible queries
- whether the object is routed
- whether the object appears redundant or unused

Why:

- realistic advisor-style introspection
- high-value diagnostic action
- helps reduce thrashing

## Workstream 4: Improve Grader Legibility

### 14. Standardize a stable score breakdown

Expose a consistent score breakdown on submit and `/grader`:

- `visible_gated_improvement`
- `holdout_gated_improvement`
- `correctness_score`
- `migration_score`
- `storage_score`
- `budget_penalty`
- `routed_query_count`
- `incorrect_query_count`
- `unused_object_count`
- `rejection_reason_histogram`

Why:

- improves trust in the grader
- helps exploit checking
- helps human review

### 15. Improve benchmark diagnostics, not just benchmark numbers

When a benchmark is poor, return more than a low reward.

Add or standardize:

- `top_rejection_reason`
- `rejection_reason_histogram`
- `best_candidate_object`
- scope-local benchmark score fields

Do not:

- reveal hidden holdout structure
- reveal the exact optimal next action

## Workstream 5: Grounded Novelty

These should come after the reward/state improvements above.

### 16. Add SLA-aware weighting

This is still the strongest novelty addition.

Add a lightweight priority tier:

- `critical`
- `high`
- `standard`

Use it to influence:

- workload weighting
- hotspot summaries
- benchmark explanations

Why this is worth doing:

- highly realistic
- easy to explain to judges
- makes the benchmark feel like a real warehouse optimization system

Important:

- tie it directly to current workload weights
- do not make it arbitrary

### 17. Add refresh / freshness realism

This is the second strongest novelty addition.

Use either:

- task-level `freshness_sensitivity`
- or cluster-level refresh tiers:
  - `batch`
  - `daily`
  - `near_realtime`

Use this to modulate:

- refresh-cost shaping
- diagnostics
- maintenance tradeoff explanations

Keep it simple and operationally plausible.

### 18. Strengthen holdout generalization through task design, not scoring hacks

Recommended:

- make hard tasks easier to overfit if the agent only optimizes visible queries narrowly
- design holdout queries so reusable objects are more successful than narrow ones

Avoid:

- difficulty-specific special-case score formulas unless absolutely necessary

Why:

- cleaner benchmark
- easier to defend
- better generalization pressure

## Changes To Avoid

- do not reward read-only browsing actions
- do not add actions that leak holdout structure
- do not add opaque or judge-unfriendly shaping heuristics
- do not bloat the action space
- do not make dense reward dominate final score
- do not add novelty that feels like a toy mechanic
- do not introduce task-specific hacks that reduce suite consistency

## Final Priority Order

## Priority 1

- environment-level step-budget termination with forced submit
- scope-local benchmark memory
- scoped benchmark reward
- utility-based create/modify/drop shaping
- soft resource-pressure shaping
- richer `SchemaOptState`
- reward decomposition in action feedback

## Priority 2

- `inspect_budget_status`
- `inspect_object_coverage`
- stable grader breakdowns
- stronger benchmark diagnostics
- typed error penalties

## Priority 3

- SLA-aware weighting
- freshness / refresh realism
- stronger hard-task holdout generalization via task design

## Suggested Implementation Sequence

### Phase 1

- fix episode boundaries
- replace global benchmark memory
- redesign create/modify/drop shaping
- add reward decomposition fields

### Phase 2

- expand state fields
- add budget and object-coverage actions
- standardize grader / benchmark diagnostics

### Phase 3

- add SLA metadata
- add freshness metadata
- refine hard-task holdout design

## Summary

Given the current baseline and the team’s parallel proposal, the best path is:

1. keep the final score unchanged
2. make intermediate reward scope-correct and denser
3. make state and diagnostics more operationally realistic
4. add grounded novelty through SLA and maintenance realism

This is the most likely route to improving the benchmark without making it less honest, less legible, or less useful.
