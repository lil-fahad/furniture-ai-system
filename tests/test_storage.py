from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from furniture_ai import storage
from furniture_ai.contracts import BookingCreate
from furniture_ai.storage import BookingStore


def booking() -> BookingCreate:
    return BookingCreate(
        customer_name="Leak Tester",
        contact="leak@example.com",
        requested_at="2026-08-10T10:00:00+03:00",
        notes="connection lifecycle",
    )


class TrackingConnection(sqlite3.Connection):
    opened = 0
    closed = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        TrackingConnection.opened += 1

    def close(self) -> None:
        TrackingConnection.closed += 1
        super().close()


@pytest.fixture
def tracked_connect(monkeypatch: pytest.MonkeyPatch):
    TrackingConnection.opened = 0
    TrackingConnection.closed = 0
    real_connect = sqlite3.connect

    def connect(path, **kwargs):
        return real_connect(path, factory=TrackingConnection, **kwargs)

    monkeypatch.setattr(storage.sqlite3, "connect", connect)
    return TrackingConnection


def test_connections_are_closed_after_each_operation(
    tmp_path: Path, tracked_connect: type[TrackingConnection]
) -> None:
    store = BookingStore(tmp_path / "bookings.sqlite3")
    assert tracked_connect.opened == 1
    assert tracked_connect.closed == 1, "connection leaked by _initialize"

    store.create(booking())
    assert tracked_connect.opened == tracked_connect.closed == 2, "connection leaked by create"

    store.list()
    assert tracked_connect.opened == tracked_connect.closed == 3, "connection leaked by list"


def test_connection_closed_and_rolled_back_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_connect = sqlite3.connect

    class FailingConnection(TrackingConnection):
        def execute(self, sql, parameters=(), /):
            if "INSERT" in sql:
                raise sqlite3.OperationalError("simulated failure")
            return super().execute(sql, parameters)

    def connect(path, **kwargs):
        return real_connect(path, factory=FailingConnection, **kwargs)

    monkeypatch.setattr(storage.sqlite3, "connect", connect)
    TrackingConnection.opened = 0
    TrackingConnection.closed = 0

    store = BookingStore(tmp_path / "bookings.sqlite3")
    with pytest.raises(sqlite3.OperationalError):
        store.create(booking())
    assert TrackingConnection.opened == TrackingConnection.closed, "connection leaked on error"
    assert store.list() == [], "failed insert must be rolled back"


def test_busy_timeout_set_on_every_connection(tmp_path: Path) -> None:
    # PRAGMA busy_timeout is per-connection; a fresh connection (not just the
    # schema-init one) must have it applied.
    store = BookingStore(tmp_path / "bookings.sqlite3")
    with closing(store._connect()) as connection:
        timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    assert timeout == 10000


def test_create_and_list_roundtrip(tmp_path: Path) -> None:
    store = BookingStore(tmp_path / "bookings.sqlite3")
    created = store.create(booking())
    listed = store.list()
    assert [item.id for item in listed] == [created.id]
    assert listed[0].status == "pending"
    assert listed[0].customer_name == "Leak Tester"
