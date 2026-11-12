"""A run's deadline: the time by which its caller stops waiting (GEO-3697).

geo-chat polls a debates.pick_moments run for a fixed time and then falls back to its
own picker, throwing away whatever the run returns later. A run that holds a
usable answer should return it before then, rather than spend the time on a
send-back or a judge whose answer nobody will read.

A `Deadline` is a point in wall-clock time. Each step that may call Claude asks it
first:

- `allows_request()`: whether a request is still worth starting (at least
  MIN_REQUEST_S left). When not, the step skips the call and keeps what it holds.
- `request_timeout(ceiling)`: the most a request may take, the time left or its
  usual ceiling, whichever is shorter, so no request runs past the deadline.

claude_json.ask_json checks both before every answer it asks for. A later call
(a judge, a claim-link call) checks `allows_request()` before it starts and
passes the deadline on to ask_json.

The clock starts when the facade enqueued the run (`ENQUEUED_AT`, in the run's
Hatchet metadata), so queue time and a Hatchet retry count against it, as they do
for geo-chat, whose clock starts once the enqueue returns. A run without the stamp
starts its clock when its attempt starts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

# The run metadata key the facade stamps with the time it enqueued a run (epoch seconds).
ENQUEUED_AT = "enqueued_at"
# The least time worth starting a Claude request with: a reel answer usually takes longer, so a
# request started with less left would most likely be cut off at the deadline, its cost spent
# for nothing.
MIN_REQUEST_S = 60


@dataclass(frozen=True)
class Deadline:
    """The wall-clock time `at` (epoch seconds) a run must be done by; `clock` reads the time now."""

    at: float
    clock: Callable[[], float] = field(default=time.time, compare=False)

    @classmethod
    def after(cls, seconds: float, *, start: Optional[float] = None, clock: Callable[[], float] = time.time) -> "Deadline":
        """`seconds` after `start`, or after now when there is no start."""
        return cls((clock() if start is None else start) + seconds, clock)

    def left(self) -> float:
        """Seconds until the deadline, never below zero."""
        return max(0.0, self.at - self.clock())

    def allows_request(self) -> bool:
        """Whether a Claude request is still worth starting (MIN_REQUEST_S left)."""
        return self.left() >= MIN_REQUEST_S

    def request_timeout(self, ceiling: float) -> float:
        """The most a request started now may take: the time left, or `ceiling` if that is less."""
        return min(ceiling, self.left())


def enqueued_at(metadata: Optional[Mapping[str, Any]], clock: Callable[[], float] = time.time) -> float:
    """When the facade enqueued the run, from its metadata; now when the stamp is missing or
    unreadable. A stamp ahead of this host's clock (skew between hosts) counts as now."""
    now = clock()
    try:
        stamped = float((metadata or {})[ENQUEUED_AT])
    except (KeyError, TypeError, ValueError):
        return now
    return min(stamped, now) if math.isfinite(stamped) else now


__all__ = ["ENQUEUED_AT", "MIN_REQUEST_S", "Deadline", "enqueued_at"]
