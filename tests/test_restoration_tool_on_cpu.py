import pytest

from verl.tools.restoration_tool import RestorationTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)


def _build_tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="restore_image",
            description="Apply an image restoration action.",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "action": OpenAIFunctionPropertySchema(
                        type="string",
                        description="The restoration action to apply.",
                    )
                },
                required=["action"],
            ),
        ),
    )


def _build_tool(**config) -> RestorationTool:
    return RestorationTool(config=config, tool_schema=_build_tool_schema())


def test_repeat_penalty_discourages_low_gain_repeats_on_cpu():
    tool = _build_tool(
        alpha=0.9,
        reward_scale=5.0,
        repeat_action_penalty=0.2,
        repeat_low_gain_penalty=0.8,
        repeat_low_gain_threshold=0.05,
    )

    fresh_reward = tool._calculate_reward(
        prev_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        curr_scores=[1.02, 1.02, 1.02, 1.02, 1.02],
        identity_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        action="ridcp",
        actions_history=[],
    )
    repeated_reward = tool._calculate_reward(
        prev_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        curr_scores=[1.02, 1.02, 1.02, 1.02, 1.02],
        identity_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        action="ridcp",
        actions_history=["ridcp"],
    )

    assert fresh_reward["repeat_penalty"] == pytest.approx(0.0)
    assert repeated_reward["repeat_penalty"] > 0.0
    assert repeated_reward["reward"] < fresh_reward["reward"]
    assert repeated_reward["consecutive_action_count"] == pytest.approx(2.0)


def test_final_iqa_v2_rewards_only_new_best_iqa_on_cpu():
    tool = _build_tool(
        reward_mode="final_iqa_v2",
        final_iqa_reward_scale=4.0,
        final_iqa_regression_penalty_scale=2.0,
        final_iqa_step_penalty=0.05,
        repeat_action_penalty=0.0,
        repeat_low_gain_penalty=0.0,
    )

    first_best = tool._calculate_reward(
        prev_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        curr_scores=[1.10, 1.10, 1.10, 1.10, 1.10],
        identity_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        action="scunet",
        actions_history=[],
        best_identity_delta=0.0,
    )
    regression = tool._calculate_reward(
        prev_scores=[1.10, 1.10, 1.10, 1.10, 1.10],
        curr_scores=[1.08, 1.08, 1.08, 1.08, 1.08],
        identity_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        action="ridcp",
        actions_history=["scunet"],
        best_identity_delta=first_best["best_identity_delta"],
    )
    new_best = tool._calculate_reward(
        prev_scores=[1.08, 1.08, 1.08, 1.08, 1.08],
        curr_scores=[1.15, 1.15, 1.15, 1.15, 1.15],
        identity_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        action="real_esrgan",
        actions_history=["scunet", "ridcp"],
        best_identity_delta=first_best["best_identity_delta"],
    )

    assert tool.suppress_tool_call_reward is True
    assert first_best["best_improvement"] == pytest.approx(0.10)
    assert first_best["reward"] == pytest.approx(0.35)
    assert regression["best_improvement"] == pytest.approx(0.0)
    assert regression["regression_penalty"] == pytest.approx(0.04)
    assert regression["reward"] == pytest.approx(-0.09)
    assert new_best["best_improvement"] == pytest.approx(0.05)
    assert new_best["reward"] == pytest.approx(0.15, abs=1e-6)


def test_stop_reward_prefers_plateau_after_good_enough_progress_on_cpu():
    tool = _build_tool(
        stop_min_step=3,
        stop_iqa_delta_threshold=0.25,
        stop_success_reward=3.0,
        stop_partial_reward=1.0,
        stop_early_penalty=-1.0,
        stop_continue_penalty=-0.5,
        stop_recent_reward_window=2,
        stop_recent_reward_threshold=0.25,
    )

    early_stop = tool._calculate_stop_reward(step=1, identity_delta=0.4, recent_rewards=[0.1])
    plateau_stop = tool._calculate_stop_reward(step=4, identity_delta=0.4, recent_rewards=[0.1, 0.0])
    premature_stop = tool._calculate_stop_reward(step=4, identity_delta=0.1, recent_rewards=[1.0, 0.8])

    assert early_stop["reward"] == pytest.approx(-1.0)
    assert plateau_stop["reward"] == pytest.approx(3.0)
    assert plateau_stop["plateau"] is True
    assert plateau_stop["good_enough"] is True
    assert premature_stop["reward"] == pytest.approx(-0.5)


def test_feedback_calls_out_repeated_low_gain_pattern_on_cpu():
    tool = _build_tool(stop_min_step=3, repeat_low_gain_threshold=0.05)

    feedback = tool._generate_feedback(
        action="ridcp",
        step=4,
        reward=0.12,
        actions_history=["ridcp", "ridcp", "ridcp", "ridcp"],
        marginal=0.01,
        identity_delta=0.35,
        consecutive_action_count=4,
    )

    assert "Consecutive uses of 'ridcp': 4" in feedback
    assert "Recent gains are small" in feedback


def test_final_iqa_v2_feedback_focuses_on_trajectory_best_on_cpu():
    tool = _build_tool(reward_mode="final_iqa_v2", stop_min_step=3)

    feedback = tool._generate_feedback(
        action="real_esrgan",
        step=3,
        reward=0.15,
        actions_history=["scunet", "ridcp", "real_esrgan"],
        marginal=0.07,
        identity_delta=0.15,
        consecutive_action_count=1,
        best_identity_delta=0.15,
        best_improvement=0.05,
        regression_penalty=0.0,
        step_penalty=0.05,
    )

    assert "Trajectory-best improvement over original image: 0.1500" in feedback
    assert "New best-IQA gain from this action: 0.0500" in feedback
    assert "This action set a new trajectory-best IQA" in feedback
    assert "Recent gains are small" not in feedback


def test_final_iqa_v2_feedback_calls_out_regression_on_cpu():
    tool = _build_tool(reward_mode="final_iqa_v2", stop_min_step=3)

    feedback = tool._generate_feedback(
        action="ridcp",
        step=4,
        reward=-0.09,
        actions_history=["scunet", "ridcp"],
        marginal=-0.02,
        identity_delta=0.08,
        consecutive_action_count=1,
        best_identity_delta=0.10,
        best_improvement=0.0,
        regression_penalty=0.04,
        step_penalty=0.05,
    )

    assert "This action fell below the trajectory-best IQA" in feedback
    assert "did not improve the trajectory-best IQA" in feedback
