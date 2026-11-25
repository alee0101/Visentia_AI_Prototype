"""Propose the next injection appointment: target = last visit + interval, ± a window.

Clinic availability is a deterministic, seedable mock: each (date, slot) is open or taken by a
hash-seeded RNG, so the same seed always yields the same calendar (reproducible demos/tests).
Weekends and holidays are closed. The proposal is always status "proposed" -- booking happens
only after physician approval (review.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Iterable

SLOT_MINUTES = 30
DEFAULT_SLOT_TIMES = tuple(time(h, m) for h in range(8, 16) for m in (0, 30) if h != 12)  # no lunch slots


def us_federal_holidays(year: int) -> set[date]:
    """US federal holidays (observed dates) for a year -- a sensible default closure list."""
    def nth_weekday(month: int, weekday: int, n: int) -> date:  # n=-1 -> last
        if n > 0:
            d = date(year, month, 1)
            return d + timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))
        d = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
        return d - timedelta(days=(d.weekday() - weekday) % 7)

    def observed(d: date) -> date:
        return d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d

    fixed = [observed(date(year, m, d)) for m, d in ((1, 1), (6, 19), (7, 4), (11, 11), (12, 25))]
    floating = [nth_weekday(1, 0, 3), nth_weekday(2, 0, 3), nth_weekday(5, 0, -1), nth_weekday(9, 0, 1),
                nth_weekday(10, 0, 2), nth_weekday(11, 3, 4)]
    return set(fixed + floating)


@dataclass
class ClinicCalendar:
    holidays: frozenset[date] = frozenset()
    slot_times: tuple[time, ...] = DEFAULT_SLOT_TIMES
    seed: int = 0
    fill_rate: float = 0.7  # fraction of slots already booked in the mock

    @classmethod
    def with_us_holidays(cls, years: Iterable[int], extra: Iterable[date] = (), **kw) -> "ClinicCalendar":
        hol = set(extra)
        for y in years:
            hol |= us_federal_holidays(y)
        return cls(holidays=frozenset(hol), **kw)

    def closed_reason(self, d: date) -> str | None:
        if d.weekday() >= 5:
            return "weekend"
        if d in self.holidays:
            return "holiday"
        return None

    def available_slots(self, d: date) -> list[datetime]:
        if self.closed_reason(d):
            return []
        rng = random.Random(f"{self.seed}:{d.isoformat()}")  # str seeds are hashed deterministically
        return [datetime.combine(d, t) for t in self.slot_times if rng.random() >= self.fill_rate]


@dataclass
class ScheduleProposal:
    last_visit: date
    interval_wk: int
    target_date: date
    window_start: date
    window_end: date
    window_days: int
    slot: datetime | None
    reason: str
    skipped: list[tuple[date, str]] = field(default_factory=list)
    status: str = "proposed"  # -> "approved" after physician sign-off

    @property
    def slot_end(self) -> datetime | None:
        return None if self.slot is None else self.slot + timedelta(minutes=SLOT_MINUTES)

    def to_dict(self) -> dict:
        return {
            "last_visit": self.last_visit.isoformat(), "interval_wk": self.interval_wk,
            "target_date": self.target_date.isoformat(), "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(), "window_days": self.window_days,
            "slot": self.slot.isoformat() if self.slot else None, "reason": self.reason,
            "skipped": [(d.isoformat(), r) for d, r in self.skipped], "status": self.status,
        }

    def allowed_dates(self) -> list[date]:
        return [self.last_visit, self.target_date, self.window_start, self.window_end] + ([self.slot.date()] if self.slot else [])


def propose_next_visit(last_visit: date, interval_wk: int, *, calendar: ClinicCalendar | None = None,
                       window_days: int = 3) -> ScheduleProposal:
    """Earliest open slot in [target - window, target + window], skipping weekends/holidays.

    Earliest (rather than closest-to-target) keeps a SHORTEN from slipping late; the window is
    the clinic's tolerance for T&E scheduling jitter. If nothing is free the slot is None and the
    reason says so -- the front desk books manually; we never silently leave the window.
    """
    cal = calendar or ClinicCalendar.with_us_holidays({last_visit.year, last_visit.year + 1})
    target = last_visit + timedelta(weeks=interval_wk)
    start, end = target - timedelta(days=window_days), target + timedelta(days=window_days)
    skipped: list[tuple[date, str]] = []
    d = start
    while d <= end:
        if why := cal.closed_reason(d):
            skipped.append((d, why))
        elif slots := cal.available_slots(d):
            off = (d - target).days
            rel = "on target" if off == 0 else f"{abs(off)} day{'s' if abs(off) > 1 else ''} {'before' if off < 0 else 'after'} target"
            reason = (f"{interval_wk}-week interval from {last_visit:%Y-%m-%d} -> target {target:%Y-%m-%d}; "
                      f"earliest open slot in ±{window_days}-day window is {slots[0]:%a %Y-%m-%d %H:%M} ({rel})")
            if skipped:
                reason += "; skipped " + ", ".join(f"{s:%Y-%m-%d} ({w})" for s, w in skipped)
            return ScheduleProposal(last_visit, interval_wk, target, start, end, window_days, slots[0], reason, skipped)
        else:
            skipped.append((d, "fully booked"))
        d += timedelta(days=1)
    reason = (f"{interval_wk}-week interval from {last_visit:%Y-%m-%d} -> target {target:%Y-%m-%d}; "
              f"no open slot in ±{window_days}-day window -- manual booking required")
    return ScheduleProposal(last_visit, interval_wk, target, start, end, window_days, None, reason, skipped)
