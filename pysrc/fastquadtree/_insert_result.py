"""InsertResult dataclass for v2.0 API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InsertResult:
    """
    Result from bulk insertion operations.

    Attributes:
        count: Number of items inserted.
        start_id: First ID in the contiguous range.
        end_id: Last ID in the contiguous range (inclusive).
    """

    count: int
    start_id: int
    end_id: int

    @property
    def ids(self) -> range:
        """Return a range of all IDs inserted."""
        return range(self.start_id, self.end_id + 1)
