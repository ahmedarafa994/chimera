from app.core.telemetry import TelemetryCollector


def test_telemetry_collector_aggregates_data() -> None:
    """Test that TelemetryCollector correctly aggregates campaign step data."""
    collector = TelemetryCollector(campaign_id="test-campaign")

    collector.record_step("persona_synthesis", {"role": "hacker", "latency": 0.5})
    collector.record_step("payload_obfuscation", {"technique": "cipher", "latency": 0.2})

    summary = collector.get_summary()
    assert summary["campaign_id"] == "test-campaign"
    assert len(summary["steps"]) == 2
    assert summary["steps"][0]["name"] == "persona_synthesis"
    assert summary["steps"][1]["data"]["technique"] == "cipher"


def test_telemetry_collector_tracks_total_latency() -> None:
    """Test that total latency is calculated correctly."""
    collector = TelemetryCollector()
    collector.record_step("step1", {"latency": 1.0})
    collector.record_step("step2", {"latency": 2.0})

    assert collector.get_total_latency() == 3.0
