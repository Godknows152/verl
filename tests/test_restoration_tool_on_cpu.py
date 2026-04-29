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