# SchemaOpt Task Reformulation Notes

This note documents the current six-task curated suite after the technical reformulation pass. The benchmark still uses the same environment contract and DuckDB execution model, but each task now carries stronger warehouse-facing identity, richer operator metadata, and clearer reuse pressure.

## Design Rules Applied

- Keep the physical benchmark contract stable: same action space, same reward/grader semantics, same visible-vs-holdout meaning.
- Add depth through workload semantics, reuse pressure, and operator-facing metadata rather than hidden score hacks.
- Preserve the existing SQL families unless a structural change is clearly justified.
- Use only additive metadata so older tasks and Spider tasks remain compatible with the loader.

## Curated Task Suite

### `schemaopt_easy_hiring_pipeline`

Technical role:
- Join-aware starter task for recruiter funnel, requisition intake, and posting hygiene reporting.

Current calibration:
- `15 visible / 6 holdout / 3 clusters / 20 max steps / 3 derived objects`
- This is the only task that received a count increase because it could support a few more grounded report variants without losing clarity.

Why this structure works:
- Candidate pipeline remains the dominant hotspot and the clearest reusable-object opportunity.
- Requisition intake and posting health add narrower but still realistic alternatives, so the agent must make a real prioritization decision.
- The extra visible queries are not just duplicates; they introduce additional segment-level variants:
  - stage-level funnel breakout
  - employment-status intake breakout
  - posting team breakout

Metadata added:
- Task-level:
  - `task_story`
  - `primary_user = recruiting operations analyst`
  - `decision_cycle = weekly_review_plus_daily_funnel_monitoring`
  - `freshness_profile = mixed_daily_and_weekly`
- Cluster-level:
  - candidate pipeline: `sla_tier=high`, `refresh_mode=daily`, `reuse_pressure=high`
  - requisition intake: `sla_tier=standard`, `refresh_mode=weekly`
  - job posting health: `sla_tier=standard`, `refresh_mode=weekly`
- Query-level:
  - `consumer_surface = recruiter_pipeline_dashboard | headcount_intake_review | job_health_audit`
  - `reuse_group` aligned by cluster
  - `report_variant_type` derived from grouping depth

Holdout intent:
- Holdout remains compact but stays in-family, so success still depends on reusable funnel design rather than memorizing one visible slice.

### `schemaopt_easy_product_adoption`

Technical role:
- Compact product telemetry task covering workspace adoption, in-app guidance, and user journey reporting.

Current calibration:
- `12 visible / 6 holdout / 3 clusters / 18 max steps / 3 derived objects`
- Count stays unchanged because this task is stronger when it remains compact and interpretable.

Why this structure works:
- The task keeps the easy tier broad without becoming noisy.
- Workspace adoption and user journey now share a conceptual reuse pathway through metadata, even though the SQL stays grounded in distinct reporting families.

Metadata added:
- Task-level:
  - `primary_user = product operations analyst`
  - `decision_cycle = daily_dashboard_review_plus_weekly_adoption_readout`
  - `freshness_profile = daily_adoption_with_weekly_behavior_rollups`
- Cluster-level:
  - workspace adoption: `sla_tier=high`, `freshness=daily`, `reuse_pressure=high`
  - in-app guidance: `sla_tier=standard`, `freshness=daily`
  - user journey: `sla_tier=standard`, `freshness=weekly`, `reuse_pressure=high`
- Query-level:
  - consumer surfaces such as `adoption_dashboard`, `onboarding_guidance_review`, and `journey_analysis_workspace`

Holdout intent:
- Holdout continues to test whether the agent chooses compact adoption-friendly objects rather than narrow visible-only slices.

### `schemaopt_medium_campaign_performance`

Technical role:
- Medium-tier hierarchy task for portfolio, campaign, ad-group, and keyword reporting.

Current calibration:
- `24 visible / 12 holdout / 4 clusters / 24 max steps / 4 derived objects`
- Structure is unchanged because this task already had sufficient room for planning and reuse.

Why this structure works:
- The reporting hierarchy now has stronger semantic identity and explicit cross-cluster reuse pressure.
- The same `reuse_group` spans portfolio, campaign, ad-group, and keyword families so the task clearly favors broader rollups.

Metadata added:
- Task-level:
  - `primary_user = paid media analytics lead`
  - `decision_cycle = daily_pacing_review_plus_weekly_channel_readout`
  - `freshness_profile = daily_performance_with_exec_rollups`
- Cluster-level:
  - portfolio health: `sla_tier=critical`, `consumer_surface=exec_dashboard`
  - campaign performance: `sla_tier=high`
  - ad group mix / keyword efficiency: `sla_tier=standard`
- Query-level:
  - `consumer_surface` distinguishes executive versus optimization workflows
  - all clusters share `reuse_group = campaign_hierarchy_rollup`

Holdout intent:
- Holdout should now be interpreted as the true test of whether the agent built a reusable hierarchy object rather than a keyword-only optimization.

### `schemaopt_medium_delivery_operations`

Technical role:
- Medium-tier PMO and delivery task across backlog execution, portfolio delivery, collaboration activity, and taxonomy governance.

Current calibration:
- `24 visible / 12 holdout / 4 clusters / 24 max steps / 4 derived objects`
- Structure stays fixed because the challenge here is semantic tradeoff, not additional volume.

Why this structure works:
- Execution backlog and portfolio delivery share a reuse pathway, while collaboration and taxonomy remain operationally important but less obviously reusable.
- This gives the task a real “broad object versus local win” decision.

Metadata added:
- Task-level:
  - `primary_user = delivery operations manager`
  - `decision_cycle = daily_delivery_standup_plus_weekly_portfolio_review`
  - `freshness_profile = daily_execution_with_weekly_governance_rollups`
- Cluster-level:
  - backlog and portfolio: `sla_tier=high`, `reuse_pressure=high`
  - collaboration: `sla_tier=standard`, `reuse_pressure=low`
  - taxonomy: `sla_tier=standard`, `reuse_pressure=medium`
- Query-level:
  - consumer surfaces distinguish standup, portfolio review, diagnostics, and governance use cases

Holdout intent:
- Holdout should reveal whether the agent picked the broader delivery-capacity object family or optimized a narrow visible cluster.

### `schemaopt_hard_mobile_revenue_ops`

Technical role:
- Flagship hard task for monetization, release, geo, device, and platform reporting.

Current calibration:
- `35 visible / 20 holdout / 5 clusters / 30 max steps / 5 derived objects`
- No structural increase was applied because this task already sits at a good planning/runtime tradeoff.

Why this structure works:
- Monetization remains the executive hotspot.
- Release, device, and platform clusters now explicitly share a distribution-oriented reuse story.
- Geo growth remains adjacent but distinct, keeping the task from collapsing into one universal object.

Metadata added:
- Task-level:
  - `primary_user = mobile revenue operations lead`
  - `decision_cycle = daily_exec_revenue_review_plus_release_monitoring`
  - `freshness_profile = mixed_daily_revenue_and_weekly_distribution_reporting`
- Cluster-level:
  - monetization yield: `sla_tier=critical`, `consumer_surface=exec_revenue_dashboard`, `refresh_mode=daily`
  - release performance: `sla_tier=high`, `refresh_mode=daily`
  - device/platform: `reuse_pressure=high`, `refresh_mode=weekly`
- Query-level:
  - monetization uses `reuse_group = revenue_yield_rollup`
  - release/device/platform share `reuse_group = distribution_mix_rollup`

Holdout intent:
- Holdout should now be read as the generalization test for cross-cluster distribution design, not just visible hotspot tuning.

### `schemaopt_hard_lifecycle_engagement`

Technical role:
- Hard lifecycle / CRM task covering campaign influence, template performance, engagement lift, churn signals, and deliverability risk.

Current calibration:
- `35 visible / 20 holdout / 5 clusters / 30 max steps / 5 derived objects`
- Structure remains unchanged because the task already has enough room for multi-step planning.

Why this structure works:
- Campaign, template, and engagement clusters now share a common lifecycle-program reuse path.
- Churn and deliverability form a distinct risk-monitoring family.
- This creates explicit fragmentation pressure: too many narrow objects should feel strategically wrong.

Metadata added:
- Task-level:
  - `primary_user = lifecycle marketing operations lead`
  - `decision_cycle = daily_risk_monitoring_plus_weekly_lifecycle_review`
  - `freshness_profile = daily_risk_signals_with_weekly_program_performance_rollups`
- Cluster-level:
  - churn signals and deliverability risk: `sla_tier=critical`, `refresh_mode=daily`
  - campaign influence and engagement lift: `sla_tier=high`
- Query-level:
  - program-facing clusters share `reuse_group = lifecycle_program_rollup`
  - risk-facing clusters share `reuse_group = risk_monitoring_rollup`

Holdout intent:
- Holdout should punish visible-only fragmentation and favor durable lifecycle object families.

## What Changed in the Runtime Surface

- `tasks.py` now loads optional task, cluster, and query metadata fields.
- `TaskSpec.task_summary()` and `reset_payload()` now include:
  - `task_story`
  - `primary_user`
  - `decision_cycle`
  - `freshness_profile`
- `ClusterSpec.to_summary()` now includes:
  - `sla_tier`
  - `business_owner`
  - `dashboard_family`
  - `refresh_mode`
  - `reuse_pressure`
- `QuerySpec.summary()` / `context()` now include:
  - `consumer_surface`
  - `latency_tier`
  - `freshness_tier`
  - `reuse_group`
  - `report_variant_type`

These additions are fully backward-compatible and do not change reward or grader behavior yet.
