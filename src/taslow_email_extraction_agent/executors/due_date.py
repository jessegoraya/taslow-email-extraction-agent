from __future__ import annotations

import re
from datetime import datetime, time, timedelta

from dateutil.parser import parse

from taslow_email_extraction_agent.agent_framework_compat import step
from taslow_email_extraction_agent.models import EmailExtractionRequest, ExtractedTaskCandidate

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


@step(name="DueDateNormalizationExecutor")
async def normalize_due_date(
    request: EmailExtractionRequest,
    task: ExtractedTaskCandidate,
) -> tuple[datetime | None, float | None, list[str]]:
    base = request.sent_date_time
    due_text = task.due_text
    if not base or not due_text:
        return None, None, []

    text = due_text.lower()
    default_due_time = time(17, 0, tzinfo=base.tzinfo)
    evidence = [
        "relative_due_date"
        if any(word in text for word in ["tomorrow", "next"])
        else "explicit_due_date"
    ]

    if "tomorrow" in text:
        due = datetime.combine(
            (base + timedelta(days=1)).date(), _extract_time(text, default_due_time)
        )
        return due, 0.86, evidence

    for weekday, target in WEEKDAYS.items():
        if weekday in text:
            days_ahead = (target - base.weekday()) % 7
            if "next" in text or days_ahead == 0:
                days_ahead = days_ahead or 7
            due = datetime.combine(
                (base + timedelta(days=days_ahead)).date(), _extract_time(text, default_due_time)
            )
            return due, 0.84, evidence

    try:
        parsed = parse(due_text, default=base)
    except (ValueError, OverflowError):
        return None, None, []

    return parsed, 0.78, evidence


def _extract_time(text: str, default_time: time) -> time:
    match = re.search(r"\b(?:at\s*)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text, re.IGNORECASE)
    if not match:
        return default_time
    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    meridiem = (match.group(3) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if not meridiem and 1 <= hour <= 7:
        hour += 12
    if hour > 23 or minute > 59:
        return default_time
    return time(hour, minute, tzinfo=default_time.tzinfo)
