"""Tests for Compass Facts API integration."""

import json
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from abevalflow.compass_facts import (
    FactPushResult,
    _build_fact_payload,
    push_gate_fact,
    push_gate_fact_from_config,
    validate_push_facts_config,
)
from abevalflow.gates.base import GateMode, GateResult, GateType
from abevalflow.schemas import PushFactsConfig


@pytest.fixture
def sample_gate_result() -> GateResult:
    """Create a sample gate result for testing."""
    return GateResult(
        gate_type=GateType.ENGINE,
        gate_name="harbor",
        passed=True,
        score=0.85,
        mode=GateMode.BLOCK,
        message="Harbor evaluation passed",
        details={"uplift": 0.15, "treatment_pass_rate": 0.9},
        evaluated_at=datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_push_facts_config() -> PushFactsConfig:
    """Create a sample push facts configuration."""
    return PushFactsConfig(
        endpoint="https://compass.stage.redhat.com/api/soundcheck/facts/",
        entity_ref="component:default/abevalflow",
        fact_ref_prefix="catalog:default/abevalflow_",
    )


class TestBuildFactPayload:
    """Tests for _build_fact_payload function."""

    def test_builds_correct_structure(self, sample_gate_result: GateResult):
        payload = _build_fact_payload(
            gate_result=sample_gate_result,
            entity_ref="component:default/test-skill",
            fact_ref="catalog:default/abevalflow_harbor",
        )

        assert "facts" in payload
        assert len(payload["facts"]) == 1

        fact = payload["facts"][0]
        assert fact["factRef"] == "catalog:default/abevalflow_harbor"
        assert fact["entityRef"] == "component:default/test-skill"
        assert "data" in fact

    def test_data_contains_gate_fields(self, sample_gate_result: GateResult):
        payload = _build_fact_payload(
            gate_result=sample_gate_result,
            entity_ref="component:default/test",
            fact_ref="catalog:default/test_harbor",
        )

        data = payload["facts"][0]["data"]
        assert data["gate_name"] == "harbor"
        assert data["passed"] is True
        assert data["score"] == 0.85
        assert data["mode"] == "block"
        assert data["message"] == "Harbor evaluation passed"
        assert "uplift" in data["details"]

    def test_evaluated_at_is_iso_format(self, sample_gate_result: GateResult):
        payload = _build_fact_payload(
            gate_result=sample_gate_result,
            entity_ref="component:default/test",
            fact_ref="catalog:default/test",
        )

        data = payload["facts"][0]["data"]
        assert data["evaluated_at"] == "2026-06-21T12:00:00+00:00"


class TestPushGateFact:
    """Tests for push_gate_fact function."""

    def test_successful_push(self, sample_gate_result: GateResult):
        with patch("abevalflow.compass_facts.urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = push_gate_fact(
                gate_result=sample_gate_result,
                endpoint="https://example.com/api/facts/",
                entity_ref="component:default/test",
                fact_ref="catalog:default/test_harbor",
            )

            assert result.success is True
            assert result.gate_name == "harbor"
            assert result.fact_ref == "catalog:default/test_harbor"
            assert result.error is None
            mock_urlopen.assert_called_once()

    def test_http_error_returns_failure(self, sample_gate_result: GateResult):
        with patch("abevalflow.compass_facts.urllib.request.urlopen") as mock_urlopen:
            import urllib.error

            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="https://example.com",
                code=500,
                msg="Internal Server Error",
                hdrs={},
                fp=None,
            )

            result = push_gate_fact(
                gate_result=sample_gate_result,
                endpoint="https://example.com/api/facts/",
                entity_ref="component:default/test",
                fact_ref="catalog:default/test_harbor",
            )

            assert result.success is False
            assert result.error == "HTTP 500: Internal Server Error"

    def test_url_error_returns_failure(self, sample_gate_result: GateResult):
        with patch("abevalflow.compass_facts.urllib.request.urlopen") as mock_urlopen:
            import urllib.error

            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            result = push_gate_fact(
                gate_result=sample_gate_result,
                endpoint="https://example.com/api/facts/",
                entity_ref="component:default/test",
                fact_ref="catalog:default/test_harbor",
            )

            assert result.success is False
            assert "Connection refused" in result.error


class TestPushGateFactFromConfig:
    """Tests for push_gate_fact_from_config function."""

    def test_builds_fact_ref_from_prefix(
        self,
        sample_gate_result: GateResult,
        sample_push_facts_config: PushFactsConfig,
    ):
        from abevalflow.compass_facts import FactPushResult

        with patch("abevalflow.compass_facts.push_gate_fact") as mock_push:
            mock_push.return_value = FactPushResult(
                gate_name="harbor",
                fact_ref="catalog:default/abevalflow_harbor",
                success=True,
            )

            result = push_gate_fact_from_config(sample_gate_result, sample_push_facts_config)

            mock_push.assert_called_once()
            call_kwargs = mock_push.call_args
            assert call_kwargs.kwargs["fact_ref"] == "catalog:default/abevalflow_harbor"
            assert result.success is True


class TestValidatePushFactsConfig:
    """Tests for validate_push_facts_config function."""

    def test_no_warning_when_config_is_none(self, caplog):
        with caplog.at_level(logging.WARNING):
            validate_push_facts_config(None, [])

        assert len(caplog.records) == 0

    def test_no_warning_when_gates_have_push_fact(
        self,
        sample_push_facts_config: PushFactsConfig,
        caplog,
    ):
        with caplog.at_level(logging.WARNING):
            validate_push_facts_config(sample_push_facts_config, ["harbor", "cisco"])

        assert len(caplog.records) == 0

    def test_warns_when_endpoint_set_but_no_gates(
        self,
        sample_push_facts_config: PushFactsConfig,
        caplog,
    ):
        with caplog.at_level(logging.WARNING):
            validate_push_facts_config(sample_push_facts_config, [])

        assert len(caplog.records) == 1
        assert "no gates have push_fact=True" in caplog.records[0].message


class TestGatePolicyPushFactMethods:
    """Tests for GatePolicy push_fact methods."""

    def test_should_push_fact_returns_false_when_no_config(self):
        from abevalflow.schemas import GatePolicy

        policy = GatePolicy()
        assert policy.should_push_fact("harbor") is False

    def test_should_push_fact_returns_false_when_gate_not_configured(
        self, sample_push_facts_config: PushFactsConfig
    ):
        from abevalflow.schemas import GatePolicy, GatePolicyItem

        policy = GatePolicy(
            push_facts=sample_push_facts_config,
            gates={"cisco": GatePolicyItem(push_fact=True)},
        )
        assert policy.should_push_fact("harbor") is False

    def test_should_push_fact_returns_true_when_configured(
        self, sample_push_facts_config: PushFactsConfig
    ):
        from abevalflow.schemas import GatePolicy, GatePolicyItem

        policy = GatePolicy(
            push_facts=sample_push_facts_config,
            gates={"harbor": GatePolicyItem(push_fact=True)},
        )
        assert policy.should_push_fact("harbor") is True

    def test_get_gates_with_push_fact(self, sample_push_facts_config: PushFactsConfig):
        from abevalflow.schemas import GatePolicy, GatePolicyItem

        policy = GatePolicy(
            push_facts=sample_push_facts_config,
            gates={
                "harbor": GatePolicyItem(push_fact=True),
                "cisco": GatePolicyItem(push_fact=True),
                "llm-review": GatePolicyItem(push_fact=False),
            },
        )

        gates = policy.get_gates_with_push_fact()
        assert set(gates) == {"harbor", "cisco"}


class TestPushFactsConfigSchema:
    """Tests for PushFactsConfig schema."""

    def test_required_fields(self):
        with pytest.raises(Exception):  # Pydantic validation error
            PushFactsConfig()

    def test_valid_config(self):
        config = PushFactsConfig(
            endpoint="https://compass.stage.redhat.com/api/soundcheck/facts/",
            entity_ref="component:default/abevalflow",
        )
        assert config.endpoint == "https://compass.stage.redhat.com/api/soundcheck/facts/"
        assert config.entity_ref == "component:default/abevalflow"
        assert config.fact_ref_prefix == "catalog:default/abevalflow_"  # default

    def test_custom_prefix(self):
        config = PushFactsConfig(
            endpoint="https://example.com/api/facts/",
            entity_ref="component:default/test",
            fact_ref_prefix="custom:ns/prefix_",
        )
        assert config.fact_ref_prefix == "custom:ns/prefix_"
