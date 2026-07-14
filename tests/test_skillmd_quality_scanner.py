"""Tests for SKILL.md quality scanner and gate."""

import json

import pytest

from abevalflow.gates.base import GateMode
from abevalflow.gates.quality import (
    SkillMdQualityGate,
    get_all_quality_gate_names,
    get_quality_gate,
)
from abevalflow.quality.skillmd_quality_scanner import (
    check_broken_references,
    check_circular_references,
    check_description_quality,
    check_file_completeness,
    check_generic_advice,
    check_imprecise_instructions,
    check_unfinished_content,
    scan_directory,
)
from abevalflow.schemas import GatePolicy, GatePolicyItem

# ---------------------------------------------------------------------------
# Description quality tests
# ---------------------------------------------------------------------------


class TestDescriptionQuality:
    def test_valid_frontmatter(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: my-skill\ndescription: A skill that does useful things\n---\n\nBody.")
        assert len(check_description_quality(p)) == 0

    def test_no_frontmatter(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("# My Skill\n\nNo frontmatter here.")
        assert any(f["rule_id"] == "quality-no-frontmatter" for f in check_description_quality(p))

    def test_missing_name(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\ndescription: A good description\n---\n\nBody.")
        assert any(f["rule_id"] == "quality-missing-name" for f in check_description_quality(p))

    def test_missing_description(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: my-skill\n---\n\nBody.")
        assert any(f["rule_id"] == "quality-missing-description" for f in check_description_quality(p))

    def test_short_description(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: my-skill\ndescription: Hi\n---\n\nBody.")
        assert any(f["rule_id"] == "quality-short-description" for f in check_description_quality(p))

    def test_description_equals_name(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: my-skill\ndescription: my-skill\n---\n\nBody.")
        assert any(f["rule_id"] == "quality-description-equals-name" for f in check_description_quality(p))

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\n: invalid: yaml: [[\n---\n\nBody.")
        assert any(f["rule_id"] == "quality-invalid-frontmatter" for f in check_description_quality(p))


# ---------------------------------------------------------------------------
# Broken references tests
# ---------------------------------------------------------------------------


class TestBrokenReferences:
    def test_broken_relative_link(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("See [guide](./nonexistent.md) for details.")
        assert any(f["rule_id"] == "quality-broken-reference" for f in check_broken_references(p))

    def test_valid_relative_link(self, tmp_path):
        (tmp_path / "guide.md").write_text("# Guide")
        p = tmp_path / "SKILL.md"
        p.write_text("See [guide](guide.md) for details.")
        assert len(check_broken_references(p)) == 0

    def test_skips_urls(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("See [docs](https://example.com) for details.")
        assert len(check_broken_references(p)) == 0

    def test_skips_anchors(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("See [section](#config) for details.")
        assert len(check_broken_references(p)) == 0

    def test_skips_templates(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("See [config](${PATH}/config.md) for details.")
        assert len(check_broken_references(p)) == 0

    def test_deduplication(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("See [a](missing.md) and [b](missing.md).")
        broken = [f for f in check_broken_references(p) if f["rule_id"] == "quality-broken-reference"]
        assert len(broken) == 1

    def test_skips_file_with_fragment(self, tmp_path):
        (tmp_path / "guide.md").write_text("# Guide")
        p = tmp_path / "SKILL.md"
        p.write_text("See [Installation](guide.md#install) for details.")
        assert len(check_broken_references(p)) == 0

    def test_skips_code_fences(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("```\nSee [link](nonexistent.md)\n```")
        assert len(check_broken_references(p)) == 0

    def test_skips_blockquotes(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("> See [link](nonexistent.md)")
        assert len(check_broken_references(p)) == 0


# ---------------------------------------------------------------------------
# File completeness tests
# ---------------------------------------------------------------------------


class TestFileCompleteness:
    def test_thin_instruction(self, tmp_path):
        (tmp_path / "instruction.md").write_text("# Task\n\nDo it.")
        assert any(f["rule_id"] == "quality-thin-instruction" for f in check_file_completeness(tmp_path))

    def test_good_instruction(self, tmp_path):
        (tmp_path / "instruction.md").write_text(
            "# Task\n\n"
            "Write a Python function that takes a list of integers and returns "
            "the sum of all even numbers in the list."
        )
        assert not any("instruction" in f["rule_id"] for f in check_file_completeness(tmp_path))

    def test_test_without_assertions(self, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_outputs.py").write_text("def test_something():\n    pass\n")
        assert any(f["rule_id"] == "quality-no-assertions" for f in check_file_completeness(tmp_path))

    def test_test_with_assertions(self, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_outputs.py").write_text("def test_something():\n    assert 1 == 1\n")
        assert not any("assertion" in f["rule_id"] for f in check_file_completeness(tmp_path))


# ---------------------------------------------------------------------------
# Imprecise instruction tests
# ---------------------------------------------------------------------------


class TestImpreciseInstructions:
    def test_hedging_detected(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\nTry to handle errors if possible.")
        findings = check_imprecise_instructions(p)
        assert any(f["rule_id"] == "quality-imprecise-instruction" for f in findings)

    def test_vague_condition_detected(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\nAdd logging as necessary.")
        findings = check_imprecise_instructions(p)
        assert any(f["rule_id"] == "quality-imprecise-instruction" for f in findings)

    def test_clean_instruction(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\nReturn the sum of all even numbers.")
        assert len(check_imprecise_instructions(p)) == 0

    def test_code_block_skipped(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\n```\ntry to do something\n```")
        assert len(check_imprecise_instructions(p)) == 0

    def test_blockquote_skipped(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\n> Try to handle this if possible.")
        assert len(check_imprecise_instructions(p)) == 0


# ---------------------------------------------------------------------------
# Unfinished content tests
# ---------------------------------------------------------------------------


class TestUnfinishedContent:
    def test_todo_detected(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\nTODO: finish this section")
        findings = check_unfinished_content(p)
        assert any(f["rule_id"] == "quality-unfinished-content" for f in findings)

    def test_fixme_detected(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\nFIXME: broken logic here")
        findings = check_unfinished_content(p)
        assert any(f["rule_id"] == "quality-unfinished-content" for f in findings)

    def test_placeholder_detected(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\n[INSERT YOUR API KEY HERE]")
        findings = check_unfinished_content(p)
        assert any(f["rule_id"] == "quality-unfinished-content" for f in findings)

    def test_tbd_detected(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\nThe output format is TBD")
        findings = check_unfinished_content(p)
        assert any(f["rule_id"] == "quality-unfinished-content" for f in findings)

    def test_clean_content(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\nThis skill analyzes CSV data.")
        assert len(check_unfinished_content(p)) == 0

    def test_code_block_skipped(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\n```\n# TODO: implement\n```")
        assert len(check_unfinished_content(p)) == 0


# ---------------------------------------------------------------------------
# Generic advice tests
# ---------------------------------------------------------------------------


class TestGenericAdvice:
    def test_best_practices_detected(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\nFollow best practices when coding.")
        findings = check_generic_advice(p)
        assert any(f["rule_id"] == "quality-generic-advice" for f in findings)

    def test_handle_errors_detected(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\nHandle errors gracefully.")
        findings = check_generic_advice(p)
        assert any(f["rule_id"] == "quality-generic-advice" for f in findings)

    def test_specific_content_clean(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\nReturn HTTP 400 for invalid input.")
        assert len(check_generic_advice(p)) == 0

    def test_code_block_skipped(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: x\ndescription: y\n---\n\n```\n# follow best practices\n```")
        assert len(check_generic_advice(p)) == 0


# ---------------------------------------------------------------------------
# scan_directory tests
# ---------------------------------------------------------------------------
# Circular references tests
# ---------------------------------------------------------------------------


class TestCircularReferences:
    def test_no_cycle(self, tmp_path):
        skills = tmp_path / "skills"
        (skills / "skill-a").mkdir(parents=True)
        (skills / "skill-b").mkdir(parents=True)
        (skills / "skill-a" / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: A\n---\n\nThis skill calls /skill-b."
        )
        (skills / "skill-b" / "SKILL.md").write_text("---\nname: skill-b\ndescription: B\n---\n\nA standalone skill.")
        assert len(check_circular_references(tmp_path)) == 0

    def test_cycle_detected(self, tmp_path):
        skills = tmp_path / "skills"
        (skills / "skill-a").mkdir(parents=True)
        (skills / "skill-b").mkdir(parents=True)
        (skills / "skill-a" / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: A\n---\n\nThis skill calls /skill-b."
        )
        (skills / "skill-b" / "SKILL.md").write_text(
            "---\nname: skill-b\ndescription: B\n---\n\nThis skill invokes /skill-a."
        )
        findings = check_circular_references(tmp_path)
        assert any(f["rule_id"] == "quality-circular-reference" for f in findings)

    def test_single_skill_no_cycle(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "SKILL.md").write_text("---\nname: my-skill\ndescription: Solo\n---\n\nA single skill.")
        assert len(check_circular_references(tmp_path)) == 0

    def test_no_skills_dir(self, tmp_path):
        assert len(check_circular_references(tmp_path)) == 0

    def test_uses_frontmatter_name(self, tmp_path):
        skills = tmp_path / "skills"
        (skills / "dir-a").mkdir(parents=True)
        (skills / "dir-b").mkdir(parents=True)
        (skills / "dir-a" / "SKILL.md").write_text("---\nname: alpha\ndescription: A\n---\n\nThis skill calls /beta.")
        (skills / "dir-b" / "SKILL.md").write_text("---\nname: beta\ndescription: B\n---\n\nThis skill invokes /alpha.")
        findings = check_circular_references(tmp_path)
        assert any(f["rule_id"] == "quality-circular-reference" for f in findings)


# ---------------------------------------------------------------------------
# scan_directory tests
# ---------------------------------------------------------------------------


class TestScanDirectory:
    def test_nonexistent_directory(self, tmp_path):
        result = scan_directory(tmp_path / "does_not_exist")
        assert result["findings"] == []

    def test_aggregates_all_checks(self, tmp_path):
        (tmp_path / "SKILL.md").write_text("---\nname: x\n---\n\nTODO: finish this.")
        (tmp_path / "instruction.md").write_text("# T\n\nDo.")
        result = scan_directory(tmp_path)
        assert len(result["findings"]) >= 2

    def test_empty_directory(self, tmp_path):
        result = scan_directory(tmp_path)
        assert result["findings"] == []

    def test_clean_submission(self, tmp_path):
        (tmp_path / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: A useful skill for data analysis\n---\n\n"
            "This skill helps analyze CSV data and produce summary statistics."
        )
        (tmp_path / "instruction.md").write_text(
            "# Analyze CSV\n\n"
            "Given a CSV file, read it with pandas, compute mean/median/std for "
            "all numeric columns, and output a summary table."
        )
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_outputs.py").write_text("def test_output():\n    assert 'mean' in result\n")
        result = scan_directory(tmp_path)
        assert len(result["findings"]) == 0


# ---------------------------------------------------------------------------
# Gate registration tests
# ---------------------------------------------------------------------------


class TestSkillMdQualityRegistration:
    def test_registered(self):
        assert "skillmd-quality" in get_all_quality_gate_names()

    def test_get_by_name(self):
        gate = get_quality_gate("skillmd-quality")
        assert isinstance(gate, SkillMdQualityGate)


# ---------------------------------------------------------------------------
# Gate evaluation tests
# ---------------------------------------------------------------------------


class TestSkillMdQualityGate:
    def test_disabled(self, tmp_path):
        policy = GatePolicy(gates={"quality": GatePolicyItem(mode=GateMode.DISABLED)})
        gate = SkillMdQualityGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is True
        assert result.mode == GateMode.DISABLED

    def test_no_scan_file(self, tmp_path):
        policy = GatePolicy()
        gate = SkillMdQualityGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is True
        assert "not_found" in result.details.get("status", "")

    def test_no_findings(self, tmp_path):
        (tmp_path / "skillmd-quality-scan.json").write_text(json.dumps({"findings": []}))
        policy = GatePolicy()
        gate = SkillMdQualityGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is True
        assert result.score == 1.0

    def test_warn_mode_always_passes(self, tmp_path):
        scan_data = {"findings": [{"severity": "high", "message": "Issue", "rule_id": "q-001"}]}
        (tmp_path / "skillmd-quality-scan.json").write_text(json.dumps(scan_data))
        policy = GatePolicy(gates={"quality": GatePolicyItem(mode=GateMode.WARN)})
        gate = SkillMdQualityGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is True

    def test_block_mode_fails_on_high(self, tmp_path):
        scan_data = {"findings": [{"severity": "high", "message": "Issue", "rule_id": "q-001"}]}
        (tmp_path / "skillmd-quality-scan.json").write_text(json.dumps(scan_data))
        policy = GatePolicy(gates={"quality": GatePolicyItem(mode=GateMode.BLOCK)})
        gate = SkillMdQualityGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is False

    def test_block_mode_missing_fails(self, tmp_path):
        policy = GatePolicy(gates={"quality": GatePolicyItem(mode=GateMode.BLOCK)})
        gate = SkillMdQualityGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is False

    def test_score_calculation(self, tmp_path):
        scan_data = {
            "findings": [
                {"severity": "medium", "message": "A", "rule_id": "q-001"},
                {"severity": "low", "message": "B", "rule_id": "q-002"},
            ]
        }
        (tmp_path / "skillmd-quality-scan.json").write_text(json.dumps(scan_data))
        policy = GatePolicy()
        gate = SkillMdQualityGate()
        result = gate.evaluate(tmp_path, policy)
        expected = (0.5 + 0.75) / 2
        assert result.score == pytest.approx(expected)
