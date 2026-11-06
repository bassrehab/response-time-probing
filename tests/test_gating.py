"""Tests for gating controllers."""

import pytest

from src.gating.controllers import BangBangController, ProportionalController


class TestBangBangController:

    def test_below_threshold(self):
        ctrl = BangBangController(threshold=0.5, on_strength=1.0)
        assert ctrl(0.0) == 0.0
        assert ctrl(0.3) == 0.0
        assert ctrl(0.49) == 0.0

    def test_at_threshold(self):
        ctrl = BangBangController(threshold=0.5, on_strength=1.0)
        assert ctrl(0.5) == 1.0

    def test_above_threshold(self):
        ctrl = BangBangController(threshold=0.5, on_strength=1.0)
        assert ctrl(0.7) == 1.0
        assert ctrl(1.0) == 1.0

    def test_custom_strength(self):
        ctrl = BangBangController(threshold=0.3, on_strength=0.8)
        assert ctrl(0.2) == 0.0
        assert ctrl(0.3) == 0.8
        assert ctrl(0.9) == 0.8


class TestProportionalController:

    def test_below_threshold(self):
        ctrl = ProportionalController(threshold=0.5, max_strength=1.0, gain=2.0)
        assert ctrl(0.0) == 0.0
        assert ctrl(0.3) == 0.0
        assert ctrl(0.49) == 0.0

    def test_proportional_scaling(self):
        ctrl = ProportionalController(threshold=0.5, max_strength=1.0, gain=2.0)
        # strength = gain * (score - threshold) = 2.0 * (0.6 - 0.5) = 0.2
        assert abs(ctrl(0.6) - 0.2) < 1e-6
        # strength = 2.0 * (0.75 - 0.5) = 0.5
        assert abs(ctrl(0.75) - 0.5) < 1e-6

    def test_clamped_at_max(self):
        ctrl = ProportionalController(threshold=0.5, max_strength=1.0, gain=2.0)
        # strength = 2.0 * (1.0 - 0.5) = 1.0 (exactly at max)
        assert abs(ctrl(1.0) - 1.0) < 1e-6
        # Even higher (shouldn't happen but test clamping)
        # strength = 2.0 * (0.9 - 0.5) = 0.8
        assert abs(ctrl(0.9) - 0.8) < 1e-6

    def test_high_gain(self):
        ctrl = ProportionalController(threshold=0.5, max_strength=1.0, gain=10.0)
        # strength = 10.0 * (0.6 - 0.5) = 1.0 (clamped)
        assert abs(ctrl(0.6) - 1.0) < 1e-6

    def test_custom_params(self):
        ctrl = ProportionalController(threshold=0.3, max_strength=0.8, gain=3.0)
        assert ctrl(0.2) == 0.0
        # strength = 3.0 * (0.5 - 0.3) = 0.6
        assert abs(ctrl(0.5) - 0.6) < 1e-6
        # strength = 3.0 * (0.7 - 0.3) = 1.2, clamped to 0.8
        assert abs(ctrl(0.7) - 0.8) < 1e-6
