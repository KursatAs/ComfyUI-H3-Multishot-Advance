# -*- coding: utf-8 -*-
"""Memory-bank data structures for H3 Multishot Advance."""


class _H3ChainBank:
    """Bounded frame bank: pinned earliest entries + recency tail.

    Uses a bounded bank policy that keeps long chains from drifting:
    `frames()` always returns the first `num_fix` entries ever added, plus the
    most recent entries, capped at `max_size` total. Conditioning on a set that
    always contains the beginning of the episode is what breaks the
    shot-to-shot feedback path - each shot is no longer a pure function of the
    one before it.
    """

    def __init__(self, num_fix=1, max_size=3):
        self.num_fix = max(0, int(num_fix))
        self.max_size = max(1, int(max_size))
        self._entries = []

    def add(self, frame):
        self._entries.append(frame)
        # prune to what frames() can ever return, so a long chain does not
        # hold every decoded frame in memory for nothing
        keep_fixed = min(self.num_fix, self.max_size)
        keep_tail = self.max_size - keep_fixed
        if len(self._entries) > keep_fixed + keep_tail:
            head = self._entries[:keep_fixed]
            # keep_tail == 0 must yield NO tail: entries[len-0:] is the whole
            # list (the slice bug that unbounded earlier bank builds)
            tail = self._entries[-keep_tail:] if keep_tail > 0 else []
            self._entries = head + tail

    def frames(self):
        fixed = self._entries[:min(self.num_fix, self.max_size)]
        tail = self._entries[len(fixed):]
        keep = self.max_size - len(fixed)
        if keep <= 0:
            return list(fixed)
        # keep_tail == 0 must yield NO tail entries: tail[-0:] is the WHOLE
        # list, which is exactly the bug that let earlier banks grow unbounded
        return list(fixed) + (list(tail[-keep:]) if keep > 0 else [])

    def latest(self):
        return self._entries[-1] if self._entries else None

    def describe(self):
        fixed = min(self.num_fix, self.max_size, len(self._entries))
        total = len(self.frames())
        return f"{total} slot(s) [{fixed} pinned + {total - fixed} recent]"
