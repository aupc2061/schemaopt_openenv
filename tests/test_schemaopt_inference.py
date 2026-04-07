import pytest

from schemaopt_inference import (
    _sanitize_filename_component,
    choose_action,
    parse_action,
    run_episode,
)


def test_parse_action_rejects_target_id_for_benchmark_cluster() -> None:
    payload = '{"operation":"benchmark_cluster","target_id":"cluster_01"}'
    action, issues = parse_action(payload)

    assert action is None
    assert any("cluster_id is required for benchmark_cluster" in issue for issue in issues)


def test_parse_action_keeps_query_id_alias_for_inspect_rewrite_status() -> None:
    payload = '{"operation":"inspect_rewrite_status","query_id":"q_1"}'
    action, issues = parse_action(payload)

    assert issues == []
    assert action is not None
    assert action.query_ids == ["q_1"]


def test_sanitize_filename_component_replaces_unsafe_chars() -> None:
    assert _sanitize_filename_component("gpt/5.4 mini") == "gpt_5.4_mini"
    assert _sanitize_filename_component("   ") == "unknown"


def test_choose_action_rejects_invalid_retry_count() -> None:
    with pytest.raises(RuntimeError, match="max_action_retries must be at least 1"):
        choose_action(
            observation=object(),
            history=[],
            step=1,
            benchmark_history=[],
            cluster_context_requests={},
            model_name="gpt-5.4-mini",
            api_base_url=None,
            max_steps=1,
            max_action_retries=0,
        )


def test_run_episode_rejects_invalid_max_steps() -> None:
    with pytest.raises(RuntimeError, match="max_steps must be at least 1"):
        run_episode(
            task_id="schemaopt_easy_hiring_pipeline",
            model_name="gpt-5.4-mini",
            api_base_url=None,
            max_steps=0,
            max_action_retries=1,
        )
