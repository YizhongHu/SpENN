"""Tests for run metadata helpers."""

from __future__ import annotations

import re

from omegaconf import OmegaConf

from spenn.training.artifacts import make_run_id, run_time_stamp
from spenn.training.run import _prepare_config


def test_run_time_stamp_uses_hh_mm_ss_format() -> None:
    stamp = run_time_stamp()

    assert re.fullmatch(r"\d{2}-\d{2}-\d{2}", stamp)


def test_make_run_id_embeds_supplied_hh_mm_ss_timestamp() -> None:
    run_id = make_run_id("hooke_multibody", run_time="12-34-56")

    assert run_id.startswith("hooke_multibody_12-34-56_")


def test_prepare_config_preserves_run_time_and_embeds_it_in_auto_run_id() -> None:
    cfg = OmegaConf.create({"run": {"id_prefix": "hooke_multibody", "time": "01-02-03"}})

    prepared = _prepare_config(cfg)

    assert prepared.run.time == "01-02-03"
    assert str(prepared.run_id).startswith("hooke_multibody_01-02-03_")


def test_prepare_config_respects_explicit_run_id_with_timestamp() -> None:
    cfg = OmegaConf.create({"run_id": "explicit_id", "run": {"time": "04-05-06"}})

    prepared = _prepare_config(cfg)

    assert prepared.run.time == "04-05-06"
    assert prepared.run_id == "explicit_id"
