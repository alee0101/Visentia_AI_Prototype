"""Numeric grounding check: every number in generated text must be traceable to the packet.

Tokens are typed so a number only grounds against facts of the same kind:
  * dates  (2026-03-03, 3/3/2026, "March 3", "3 March 2026") -> allowed calendar dates
  * times  (9:30)                                             -> allowed slot times
  * pct    (8%, 8 percent)                                    -> percentage facts
  * plain  (296, 0.42, q8w -> 8)                              -> plain numeric facts
A token grounds if it is a rounding of a fact at the precision it is written with
(296 grounds 295.6; 0.4 grounds 0.42; 300 does NOT ground 296). Signs are ignored ("-8%" vs
"8% below"). Leading list markers ("1. ") are skipped. Number *words* ("two weeks") are not
checked -- the prompts require digits, and this is a documented limitation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Iterable

from .packet import VisitPacket

_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december")
_MON = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
_TOKEN = re.compile(
    rf"""(?P<iso>\b\d{{4}}-\d{{2}}-\d{{2}}\b)
    |(?P<slash>\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?\b)
    |(?P<mdy>\b(?:{_MON})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b)
    |(?P<dmy>\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MON})\b\.?(?:,?\s+\d{{4}})?)
    |(?P<time>\b\d{{1,2}}:\d{{2}}\b)
    |(?P<num>\d+(?:\.\d+)?)(?P<pct>\s*(?:%|percent\b))?""",
    re.IGNORECASE | re.VERBOSE,
)
_LIST_MARKER = re.compile(r"^\s*\d{1,2}[.)]\s", re.MULTILINE)


@dataclass
class GroundingReport:
    ok: bool
    checked: list[str] = field(default_factory=list)  # every numeric/date span examined
    ungrounded: list[dict] = field(default_factory=list)  # {"span", "start", "end", "kind"}
    issues: list[str] = field(default_factory=list)  # non-numeric problems (finality wording, readability)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checked": self.checked, "ungrounded": self.ungrounded, "issues": self.issues}

    def feedback(self) -> str:
        """Human/LLM-readable explanation used for the single retry."""
        parts = []
        if self.ungrounded:
            spans = ", ".join(f"'{u['span']}'" for u in self.ungrounded)
            parts.append(f"These numbers/dates are not in FACTS and must be removed or corrected: {spans}.")
        parts += self.issues
        return " ".join(parts)


@dataclass
class FactPool:
    plain: set[float] = field(default_factory=set)
    pct: set[float] = field(default_factory=set)
    dates: set[date] = field(default_factory=set)
    times: set[tuple[int, int]] = field(default_factory=set)
    literals: set[str] = field(default_factory=set)  # digit runs allowed verbatim (study id)

    def add_text(self, text: str) -> None:
        """Every number appearing in trusted text (e.g. the engine trace) is grounded."""
        for m in _TOKEN.finditer(text):
            if m.group("num"):
                (self.pct if m.group("pct") else self.plain).add(abs(float(m.group("num"))))
            elif m.group("slash"):
                self.plain.update(float(x) for x in m.group("slash").split("/"))


def build_fact_pool(packet: VisitPacket, *, dates: Iterable[date] = (), times: Iterable[time | datetime] = (),
                    numbers: Iterable[float] = ()) -> FactPool:
    """All facts derivable from the packet (+ schedule dates/times/numbers passed by the caller)."""
    p = FactPool()
    vals = [packet.cst_um, packet.ref_cst_um, packet.cst_sd_um or None, packet.bcva_letters,
            packet.best_bcva_letters, packet.bcva_loss_letters, packet.visit, packet.current_interval_wk,
            packet.next_interval_wk, packet.interval_change_wk, packet.ratio, *packet.fluid_probs.values(), *numbers]
    if packet.ref_cst_um is not None:
        vals.append(packet.cst_um - packet.ref_cst_um)
    if packet.cst_sd_um:
        vals.append(1.5 * packet.cst_sd_um)
    p.plain.update(abs(float(v)) for v in vals if v is not None)
    pcts = [packet.cst_change_pct, None if packet.ratio is None else packet.ratio * 100,
            *(v * 100 for v in packet.fluid_probs.values())]
    p.pct.update(abs(float(v)) for v in pcts if v is not None)
    for t in packet.trace:
        p.add_text(t)
    p.literals.update(re.findall(r"\d+", packet.study_id))
    if packet.visit_date:
        p.dates.add(packet.visit_date)
    p.dates.update(dates)
    p.times.update((t.hour, t.minute) for t in times)
    return p


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def _num_grounded(s: str, pool: set[float]) -> bool:
    x, tol = abs(float(s)), 0.5 * 10 ** -_decimals(s) + 1e-9
    return any(abs(x - f) <= tol for f in pool)


def _parse_month(name: str) -> int:
    name = name.lower().rstrip(".")
    return next(i + 1 for i, m in enumerate(_MONTHS) if m.startswith(name[:3]))


def _date_grounded(month: int, day: int, year: int | None, allowed: set[date]) -> bool:
    return any(d.month == month and d.day == day and (year is None or d.year == year) for d in allowed)


def _check_token(m: re.Match, pool: FactPool) -> bool:
    kind = m.lastgroup if m.lastgroup != "pct" else "num"
    s = m.group(0)
    if kind == "iso":
        try:
            return date.fromisoformat(s) in pool.dates
        except ValueError:
            return False
    if kind == "slash":  # a date (3/3/2026) or a fraction from the trace ("dose 4/4")
        parts = [int(x) for x in s.split("/")]
        year = None if len(parts) == 2 else (parts[2] + 2000 if parts[2] < 100 else parts[2])
        if 1 <= parts[0] <= 12 and 1 <= parts[1] <= 31 and _date_grounded(parts[0], parts[1], year, pool.dates):
            return True
        return all(_num_grounded(x, pool.plain) for x in s.split("/"))
    if kind in ("mdy", "dmy"):
        mon = re.search(rf"\b({_MON})\b", s, re.I).group(1)
        nums = [int(x) for x in re.findall(r"\d+", s)]
        day = nums[0]
        year = next((n for n in nums[1:] if n > 31), None)
        return _date_grounded(_parse_month(mon), day, year, pool.dates)
    if kind == "time":
        h, mi = (int(x) for x in s.split(":"))
        return (h, mi) in pool.times or (h + 12, mi) in pool.times
    num = m.group("num")
    if num in pool.literals:
        return True
    return _num_grounded(num, pool.pct if m.group("pct") else pool.plain)


def check_grounding(text: str, pool: FactPool) -> GroundingReport:
    """Extract every number/date/time span in `text` and verify it against `pool`."""
    skip = {m.start() + len(m.group(0)) - len(m.group(0).lstrip()) for m in _LIST_MARKER.finditer(text)}
    rep = GroundingReport(ok=True)
    for m in _TOKEN.finditer(text):
        if m.start() in skip:
            continue
        rep.checked.append(m.group(0))
        if not _check_token(m, pool):
            rep.ungrounded.append({"span": m.group(0), "start": m.start(), "end": m.end(), "kind": m.lastgroup})
    rep.ok = not rep.ungrounded
    return rep
