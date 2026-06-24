"""Timezone and attempt-id helpers for staged studies."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULT_STUDY_TIMEZONE = "America/New_York"


def resolve_timezone(name: str | None = None) -> ZoneInfo:
    """Return the study timezone, defaulting to America/New_York."""

    return ZoneInfo(name or DEFAULT_STUDY_TIMEZONE)


STUDY_TIMEZONE = resolve_timezone()


def new_attempt_id(moment: datetime | None = None, *, tz: ZoneInfo = STUDY_TIMEZONE) -> str:
    """Return an attempt id of the form ``YYYYMMDDTHHMMSS-0400`` in ``tz``."""

    moment = moment or datetime.now(tz)
    return moment.astimezone(tz).strftime("%Y%m%dT%H%M%S%z")
