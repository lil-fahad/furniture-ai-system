from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from furniture_ai.contracts import Booking, BookingCreate


class BookingStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        # busy_timeout is per-connection, so every connection (not just the
        # one used for schema init) must set it to survive concurrent access.
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                    CREATE TABLE IF NOT EXISTS bookings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        customer_name TEXT NOT NULL,
                        contact TEXT NOT NULL,
                        requested_at TEXT NOT NULL,
                        notes TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
            )

    def ping(self) -> None:
        """Verify that SQLite is reachable and able to execute a query."""
        with closing(self._connect()) as connection:
            connection.execute("SELECT 1").fetchone()

    def create(self, request: BookingCreate) -> Booking:
        created_at = datetime.now(UTC).isoformat()
        with self._lock, closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO bookings
                    (customer_name, contact, requested_at, notes, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.customer_name,
                        request.contact,
                        request.requested_at,
                        request.notes,
                        "pending",
                        created_at,
                    ),
                )
            booking_id = int(cursor.lastrowid)
        return Booking(
            id=booking_id,
            status="pending",
            created_at=created_at,
            **request.model_dump(),
        )

    def list(self, limit: int = 100) -> list[Booking]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM bookings ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Booking.model_validate(dict(row)) for row in rows]
