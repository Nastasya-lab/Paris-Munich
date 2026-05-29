from weather_tmax_bot.operations.launch_readiness import assess_launch_readiness


def test_launch_readiness_returns_forward_and_outcome_flags():
    readiness = assess_launch_readiness()

    assert "ready_for_forward_ops" in readiness
    assert "ready_for_outcome_monitoring" in readiness
    assert "blocking_reasons" in readiness
    assert "next_action" in readiness
