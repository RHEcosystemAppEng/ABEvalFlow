"""Tests for scripts/backfill_scorecards.py — checkpoint logic and BackfillState."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.backfill_scorecards import BackfillState, _load_checkpoint, _save_checkpoint


class TestBackfillState:
    def test_default_values(self) -> None:
        state = BackfillState()
        assert state.last_processed_key == ""
        assert state.processed_count == 0
        assert state.skipped_count == 0
        assert state.error_count == 0
        assert state.errors == []

    def test_with_values(self) -> None:
        state = BackfillState(
            last_processed_key="prefix/scorecard.json",
            processed_count=10,
            skipped_count=3,
            error_count=1,
            errors=[{"key": "bad.json", "error": "parse error"}],
        )
        assert state.processed_count == 10
        assert state.error_count == 1


class TestCheckpoint:
    def test_save_and_load(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "checkpoint.json"
        state = BackfillState(
            last_processed_key="20260101_sample/scorecard.json",
            processed_count=5,
            skipped_count=2,
            error_count=1,
            errors=[{"key": "bad.json", "error": "fail"}],
        )
        _save_checkpoint(ckpt, state)

        loaded = _load_checkpoint(ckpt)
        assert loaded.last_processed_key == "20260101_sample/scorecard.json"
        assert loaded.processed_count == 5
        assert loaded.skipped_count == 2
        assert loaded.error_count == 1
        assert len(loaded.errors) == 1

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "missing.json"
        loaded = _load_checkpoint(ckpt)
        assert loaded.processed_count == 0
        assert loaded.last_processed_key == ""

    def test_save_truncates_errors(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "checkpoint.json"
        state = BackfillState(
            error_count=100,
            errors=[{"key": f"key-{i}", "error": "fail"} for i in range(100)],
        )
        _save_checkpoint(ckpt, state, max_errors=10)

        data = json.loads(ckpt.read_text())
        assert len(data["errors"]) == 10
        assert data["errors"][0]["key"] == "key-90"

    def test_checkpoint_round_trip_preserves_counts(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "checkpoint.json"
        original = BackfillState(
            last_processed_key="key-50",
            processed_count=50,
            skipped_count=20,
            error_count=5,
            errors=[{"key": f"err-{i}", "error": "oops"} for i in range(5)],
        )
        _save_checkpoint(ckpt, original)
        loaded = _load_checkpoint(ckpt)
        assert loaded.processed_count == original.processed_count
        assert loaded.skipped_count == original.skipped_count
        assert loaded.error_count == original.error_count
