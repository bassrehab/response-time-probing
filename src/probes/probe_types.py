"""Probe type definitions."""

from enum import Enum


class ProbeType(str, Enum):
    JAILBREAK = "jailbreak"
    HARMFUL = "harmful"
    SYCOPHANCY = "sycophancy"
    SANDBAGGING = "sandbagging"
