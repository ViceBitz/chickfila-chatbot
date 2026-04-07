import re
from typing import Any


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_integer(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = normalize_text(value)
    text = re.sub(r"[^0-9+-]", "", text)
    if not text or text in {"-", "+"}:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def normalize_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_text(value).replace(",", ".")
    text = re.sub(r"[^0-9+\-.]", "", text)
    if not text or text in {"-", "+", ".", "-.", "+."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_list_of_text(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = normalize_text(item)
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return normalized


def normalize_hours_entry(entry: dict) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    day = entry.get("dayOfWeek", entry.get("day_of_week", ""))
    if isinstance(day, list):
        day = [normalize_text(item) for item in day if normalize_text(item)]
    else:
        day = normalize_text(day)
    return {
        "day_of_week": day,
        "opens": normalize_text(entry.get("opens")),
        "closes": normalize_text(entry.get("closes")),
    }


def normalize_address(address: Any) -> dict[str, str]:
    if not isinstance(address, dict):
        address = {}
    return {
        "street": normalize_text(address.get("street") or address.get("streetAddress")),
        "city": normalize_text(address.get("city") or address.get("addressLocality")),
        "state": normalize_text(address.get("state") or address.get("addressRegion")),
        "zip": normalize_text(address.get("zip") or address.get("postalCode")),
        "country": normalize_text(address.get("country") or address.get("addressCountry")) or "US",
    }
