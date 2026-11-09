"""Hivatalos közlemények a sokk vagy az ütemezett esemény címkézéséhez (spec/07, 3.; docs/hirek.md).

Csak akkor fut, ha az `S` napon piaci sokk-jel volt, vagy ütemezett esemény
esett rá. Nem jelez, nem pontoz és nem jósol: megmutatja, mit tett közzé
aznap a kiadó. Cím, forrás, időpont és link — a szöveget nem tároljuk.

Forrás csak az, ami bot-ellenőrzés nélkül, nyilvánosan elérhető. A BLS
hírcsatornája gépi lekérést tilt (403): nem kerüljük meg, kimarad.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import requests

from pipeline import log as logging_setup
from pipeline.events.calendar import EVENTS

log = logging_setup.get_logger(__name__)

USER_AGENT = "Calyze research (github.com/sigmundsabo/calyze-data)"
FEEDS: tuple[tuple[str, str, str], ...] = (
    ("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml", "www.federalreserve.gov"),
    ("European Central Bank", "https://www.ecb.europa.eu/rss/press.html", "www.ecb.europa.eu"),
)
MAX_ITEMS = 10


def parse_feed(xml_text: str, source: str, host: str) -> list[dict[str, str]]:
    """RSS 2.0 → tételek. Csak a kiadó saját domainjére mutató https-link marad."""
    out: list[dict[str, str]] = []
    # Az RSS-nek nem kell DTD: ami entitást vagy DOCTYPE-ot hoz, azt el sem
    # kezdjük értelmezni (entitás-robbanás és külső entitás ellen).
    head = xml_text[:4096].upper()
    if "<!DOCTYPE" in head or "<!ENTITY" in xml_text.upper():
        return out
    root = ET.fromstring(xml_text)  # noqa: S314 — DTD nélküli bemenet, lásd fent
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        raw_date = (item.findtext("pubDate") or "").strip()
        if not title or not link or not raw_date:
            continue
        parsed = urlparse(link)
        if parsed.scheme != "https" or parsed.netloc != host:
            continue
        try:
            published = parsedate_to_datetime(raw_date).astimezone(UTC)
        except (TypeError, ValueError):
            continue
        out.append({"source": source, "title": title[:200], "link": link, "published": published.isoformat()})
    return out


def triggers(session: date, market_signals: list[str]) -> list[str]:
    """Mi indokolja a lekérést: piaci sokk-jel vagy az `S` napra eső esemény."""
    reasons = [f"shock:{s}" for s in market_signals]
    reasons += [f"event:{e.id}" for e in EVENTS if e.day == session]
    return reasons


def in_window(items: list[dict[str, str]], session: date) -> list[dict[str, str]]:
    """Az `S` nap és az előtte lévő nap közleményei (UTC), a legfrissebb elöl."""
    start = datetime(session.year, session.month, session.day, tzinfo=UTC) - timedelta(days=1)
    end = start + timedelta(days=2, hours=6)
    kept = [i for i in items if start <= datetime.fromisoformat(i["published"]) < end]
    return sorted(kept, key=lambda i: i["published"], reverse=True)[:MAX_ITEMS]


def official_news(
    session: date,
    market_signals: list[str],
    fetch: Callable[[str], Any] | None = None,
) -> dict[str, object]:
    reasons = triggers(session, market_signals)
    if not reasons:
        return {"status": "no_trigger"}
    get = fetch or (lambda url: requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20))
    items: list[dict[str, str]] = []
    failed: list[str] = []
    for source, url, host in FEEDS:
        try:
            response = get(url)
            if response.status_code != 200:
                failed.append(source)
                continue
            # bájtként: a Fed csatornája BOM-mal kezdődik, és nem mond karakterkészletet
            items += parse_feed(response.content.decode("utf-8-sig", errors="replace"), source, host)
        except Exception as error:  # noqa: BLE001 — egy forrás kiesése nem állíthatja meg a közzétételt
            log.info("news_feed_failed", source=source, error=type(error).__name__)
            failed.append(source)
    kept = in_window(items, session)
    log.info("news_labelled", triggers=len(reasons), items=len(kept), failed=len(failed))
    return {
        "status": "ok",
        "session": session.isoformat(),
        "triggers": reasons,
        "items": kept,
        "failed_sources": failed,
    }
