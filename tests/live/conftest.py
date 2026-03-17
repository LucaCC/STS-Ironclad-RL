"""Shared fixtures for fast, deterministic live-pipeline tests."""

from __future__ import annotations

import pytest

from .factories import (
    encode_snapshot,
    make_card,
    make_combat_snapshot,
    make_enemy,
    make_event_snapshot,
)


@pytest.fixture
def combat_snapshot_factory():
    return make_combat_snapshot


@pytest.fixture
def event_snapshot_factory():
    return make_event_snapshot


@pytest.fixture
def card_factory():
    return make_card


@pytest.fixture
def enemy_factory():
    return make_enemy


@pytest.fixture
def observation_encoder():
    return encode_snapshot
