"""Tests for security and quality gate registries."""

import json

import pytest

from abevalflow.gates.base import GateMode, GateType, Severity
from abevalflow.gates.quality import (
    LLMReviewGate,
    get_all_quality_gate_names,
    get_all_quality_gates,
    get_quality_gate,
)
from abevalflow.gates.security import (
    CiscoGate,
    get_all_security_gate_names,
    get_all_security_gates,
    get_security_gate,
)
from abevalflow.schemas import GatePolicy, GatePolicyItem


class TestSecurityGateRegistry:
    """Tests for security gate registration and lookup."""

    def test_get_all_security_gate_names(self):
        names = get_all_security_gate_names()
        assert "cisco" in names

    def test_get_all_security_gates(self):
        gates = get_all_security_gates()
        assert len(gates) >= 1
        assert any(g.name == "cisco" for g in gates)

    def test_get_security_gate_cisco(self):
        gate = get_security_gate("cisco")
        assert isinstance(gate, CiscoGate)
        assert gate.name == "cisco"

    def test_get_security_gate_unknown(self):
        with pytest.raises(KeyError):
            get_security_gate("unknown_gate")


class TestCiscoGate:
    """Tests for Cisco security gate."""

    def test_evaluate_disabled(self, tmp_path):
        policy = GatePolicy(gates={"security": GatePolicyItem(mode=GateMode.DISABLED)})
        gate = CiscoGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.gate_type == GateType.SECURITY
        assert result.gate_name == "security"
        assert result.policy_key == "cisco"
        assert result.details["scanner"] == "cisco"
        assert result.passed is True
        assert result.mode == GateMode.DISABLED
        assert "disabled" in result.message.lower()

    def test_evaluate_no_scan_file(self, tmp_path):
        policy = GatePolicy()
        gate = CiscoGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.passed is True
        assert "not_found" in result.details.get("status", "")

    def test_evaluate_no_findings(self, tmp_path):
        scan_data = {"findings": []}
        (tmp_path / "security-scan.json").write_text(json.dumps(scan_data))

        policy = GatePolicy()
        gate = CiscoGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.passed is True
        assert result.score == 1.0
        assert len(result.findings) == 0

    def test_evaluate_with_findings_warn_mode(self, tmp_path):
        scan_data = {
            "findings": [
                {"severity": "high", "message": "SQL injection", "rule_id": "SEC001", "file_path": "db.py"},
                {"severity": "medium", "message": "Unused variable", "rule_id": "SEC002"},
            ]
        }
        (tmp_path / "security-scan.json").write_text(json.dumps(scan_data))

        policy = GatePolicy(gates={"security": GatePolicyItem(mode=GateMode.WARN)})
        gate = CiscoGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.passed is True  # Warn mode always passes
        assert result.mode == GateMode.WARN
        assert len(result.findings) == 2

    def test_evaluate_with_high_findings_block_mode(self, tmp_path):
        scan_data = {
            "findings": [
                {"severity": "high", "message": "SQL injection", "rule_id": "SEC001"},
            ]
        }
        (tmp_path / "security-scan.json").write_text(json.dumps(scan_data))

        policy = GatePolicy(gates={"security": GatePolicyItem(mode=GateMode.BLOCK)})
        gate = CiscoGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.passed is False  # Block mode fails on HIGH
        assert result.mode == GateMode.BLOCK

    def test_evaluate_with_critical_finding(self, tmp_path):
        scan_data = {
            "findings": [
                {"severity": "critical", "message": "Remote code execution", "rule_id": "SEC999"},
            ]
        }
        (tmp_path / "security-scan.json").write_text(json.dumps(scan_data))

        policy = GatePolicy(gates={"security": GatePolicyItem(mode=GateMode.BLOCK)})
        gate = CiscoGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.passed is False
        assert result.findings[0].severity == Severity.CRITICAL

    def test_evaluate_only_low_findings_block_mode(self, tmp_path):
        scan_data = {
            "findings": [
                {"severity": "low", "message": "Minor issue", "rule_id": "SEC100"},
                {"severity": "info", "message": "FYI", "rule_id": "SEC101"},
            ]
        }
        (tmp_path / "security-scan.json").write_text(json.dumps(scan_data))

        policy = GatePolicy(gates={"security": GatePolicyItem(mode=GateMode.BLOCK)})
        gate = CiscoGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.passed is True  # No HIGH or CRITICAL


class TestQualityGateRegistry:
    """Tests for quality gate registration and lookup."""

    def test_get_all_quality_gate_names(self):
        names = get_all_quality_gate_names()
        assert "llm-review" in names

    def test_get_all_quality_gates(self):
        gates = get_all_quality_gates()
        assert len(gates) >= 1
        assert any(g.name == "llm-review" for g in gates)

    def test_get_quality_gate_llm_review(self):
        gate = get_quality_gate("llm-review")
        assert isinstance(gate, LLMReviewGate)
        assert gate.name == "llm-review"

    def test_get_quality_gate_unknown(self):
        with pytest.raises(KeyError):
            get_quality_gate("unknown_gate")


class TestLLMReviewGate:
    """Tests for LLM review quality gate."""

    def test_evaluate_disabled(self, tmp_path):
        policy = GatePolicy(gates={"quality": GatePolicyItem(mode=GateMode.DISABLED)})
        gate = LLMReviewGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.gate_type == GateType.QUALITY
        assert result.gate_name == "quality"
        assert result.policy_key == "llm-review"
        assert result.details["reviewer"] == "llm-review"
        assert result.passed is True
        assert result.mode == GateMode.DISABLED

    def test_evaluate_no_review_file(self, tmp_path):
        policy = GatePolicy()
        gate = LLMReviewGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.passed is True
        assert "not_found" in result.details.get("status", "")

    def test_evaluate_pass_recommendation(self, tmp_path):
        review_data = {
            "dimensions": {
                "coherence": {"score": 0.8, "finding": "Good coherence"},
                "coverage": {"score": 0.7, "finding": "Adequate coverage"},
                "clarity": {"score": 0.9, "finding": "Very clear"},
                "feasibility": {"score": 0.75, "finding": "Feasible"},
                "robustness": {"score": 0.65, "finding": "OK robustness"},
            },
            "overall_score": 0.76,
            "recommendation": "pass",
            "summary": "Overall good submission",
        }
        (tmp_path / "_ai_review.json").write_text(json.dumps(review_data))

        policy = GatePolicy()
        gate = LLMReviewGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.passed is True
        assert result.score == 0.76
        assert result.details["recommendation"] == "pass"

    def test_evaluate_fail_recommendation(self, tmp_path):
        review_data = {
            "dimensions": {
                "coherence": {"score": 0.3, "finding": "Poor coherence"},
                "coverage": {"score": 0.4, "finding": "Low coverage"},
                "clarity": {"score": 0.5, "finding": "Unclear"},
                "feasibility": {"score": 0.3, "finding": "Not feasible"},
                "robustness": {"score": 0.2, "finding": "Very weak"},
            },
            "overall_score": 0.34,
            "recommendation": "fail",
            "summary": "Major issues",
        }
        (tmp_path / "_ai_review.json").write_text(json.dumps(review_data))

        policy = GatePolicy()
        gate = LLMReviewGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.passed is False
        assert len(result.findings) > 0  # Should have findings for low scores

    def test_evaluate_warn_recommendation_block_mode(self, tmp_path):
        review_data = {
            "dimensions": {
                "coherence": {"score": 0.55, "finding": "OK coherence"},
                "coverage": {"score": 0.5, "finding": "Marginal coverage"},
                "clarity": {"score": 0.6, "finding": "Clear enough"},
                "feasibility": {"score": 0.55, "finding": "Probably feasible"},
                "robustness": {"score": 0.5, "finding": "Weak"},
            },
            "overall_score": 0.54,
            "recommendation": "warn",
            "summary": "Marginal submission",
        }
        (tmp_path / "_ai_review.json").write_text(json.dumps(review_data))

        policy = GatePolicy(gates={"quality": GatePolicyItem(mode=GateMode.BLOCK, threshold=0.6)})
        gate = LLMReviewGate()
        result = gate.evaluate(tmp_path, policy)

        # In block mode with threshold 0.6, score 0.54 should fail
        assert result.passed is False
        assert result.threshold == 0.6

    def test_evaluate_warn_recommendation_warn_mode(self, tmp_path):
        review_data = {
            "dimensions": {
                "coherence": {"score": 0.55, "finding": "OK"},
            },
            "overall_score": 0.55,
            "recommendation": "warn",
            "summary": "Marginal",
        }
        (tmp_path / "_ai_review.json").write_text(json.dumps(review_data))

        policy = GatePolicy(gates={"quality": GatePolicyItem(mode=GateMode.WARN)})
        gate = LLMReviewGate()
        result = gate.evaluate(tmp_path, policy)

        # In warn mode, warn recommendation passes
        assert result.passed is True

    def test_block_mode_fail_recommendation_high_score(self, tmp_path):
        """In BLOCK mode, score threshold is authoritative - fail recommendation can be overridden."""
        review_data = {
            "dimensions": {
                "coherence": {"score": 0.7, "finding": "Good"},
                "coverage": {"score": 0.65, "finding": "Acceptable"},
            },
            "overall_score": 0.65,  # Above default threshold of 0.6
            "recommendation": "fail",  # LLM says fail
            "summary": "Some concerns",
        }
        (tmp_path / "_ai_review.json").write_text(json.dumps(review_data))

        policy = GatePolicy(gates={"quality": GatePolicyItem(mode=GateMode.BLOCK, threshold=0.6)})
        gate = LLMReviewGate()
        result = gate.evaluate(tmp_path, policy)

        # In BLOCK mode, 0.65 >= 0.6 threshold means PASS despite fail recommendation
        assert result.passed is True
        assert "upstream: fail" in result.message

    def test_block_mode_pass_recommendation_low_score(self, tmp_path):
        """In BLOCK mode, score below threshold fails even if recommendation is pass."""
        review_data = {
            "dimensions": {
                "coherence": {"score": 0.5, "finding": "OK"},
            },
            "overall_score": 0.5,  # Below threshold
            "recommendation": "pass",  # LLM says pass
            "summary": "Looks fine",
        }
        (tmp_path / "_ai_review.json").write_text(json.dumps(review_data))

        policy = GatePolicy(gates={"quality": GatePolicyItem(mode=GateMode.BLOCK, threshold=0.6)})
        gate = LLMReviewGate()
        result = gate.evaluate(tmp_path, policy)

        # In BLOCK mode, 0.5 < 0.6 threshold means FAIL despite pass recommendation
        assert result.passed is False
        assert "upstream: pass" in result.message

    def test_block_mode_missing_artifact_fails(self, tmp_path):
        """In BLOCK mode, missing _ai_review.json fails the gate."""
        policy = GatePolicy(gates={"quality": GatePolicyItem(mode=GateMode.BLOCK)})
        gate = LLMReviewGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.passed is False
        assert "required in block mode" in result.message


class TestCiscoGateBlockMode:
    """Tests for CiscoGate fail-closed behavior in BLOCK mode."""

    def test_block_mode_missing_artifact_fails(self, tmp_path):
        """In BLOCK mode, missing security-scan.json fails the gate."""
        policy = GatePolicy(gates={"security": GatePolicyItem(mode=GateMode.BLOCK)})
        gate = CiscoGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.passed is False
        assert "required in block mode" in result.message

    def test_warn_mode_missing_artifact_passes(self, tmp_path):
        """In WARN mode, missing security-scan.json passes the gate."""
        policy = GatePolicy(gates={"security": GatePolicyItem(mode=GateMode.WARN)})
        gate = CiscoGate()
        result = gate.evaluate(tmp_path, policy)

        assert result.passed is True
        assert "scan may have been skipped" in result.message
