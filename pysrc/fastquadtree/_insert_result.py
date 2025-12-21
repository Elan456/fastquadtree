"""InsertResult dataclass for bulk insertion operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InsertResult:
    """
    Result returned from bulk insertion operations.

    This dataclass contains information about items inserted by `insert_many` or
    `insert_many_np` methods, including the total count and the contiguous range
    of auto-assigned IDs.

    Attributes:
        count: Number of items successfully inserted.
        start_id: First ID in the contiguous ID range.
        end_id: Last ID in the contiguous ID range (inclusive).

    Example:
        ```python
        result = qt.insert_many([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
        print(f"Inserted {result.count} items")
        print(f"IDs: {list(result.ids)}")  # [0, 1, 2]
        ```
    """

    count: int
    start_id: int
    end_id: int

    @property
    def ids(self) -> range:
        """
        Return a range object representing all inserted IDs.

        Returns:
            Range from start_id to end_id (inclusive).
        """
        return range(self.start_id, self.end_id + 1)
