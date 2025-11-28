#!/usr/bin/env python3
"""
QuantraCore Apex — UI Button Double-Test Harness
Author: Lamont Labs (Jesse J. Lamont)

Purpose
-------
Brute-force test EVERY (non-obviously-destructive) button in the ApexDesk UI,
and click each of them TWICE to catch flaky behaviour.

What it does
------------
1) Waits for the UI to be reachable (default: http://localhost:5000).
2) Opens a real Chromium browser (headed by default, so you can watch if you want).
3) On the initial view:
      - Finds all button-like elements.
      - Clicks each button twice, logging success/failure.
4) Walks navigation links (sidebar/nav/header) to visit other views.
      - On each view, again finds all button-like elements and double-tests them.
5) Writes:
      - Human log      → logs/ui_buttons_double/<timestamp>/button_double_test_log.txt
      - Machine summary → logs/ui_buttons_double/<timestamp>/button_double_test_summary.json
      - Per-button details → logs/ui_buttons_double/<timestamp>/button_double_test_details.json

Safety
------
To avoid destructive behaviour during testing, this harness SKIPS buttons whose
label contains any of the following (case-insensitive):

    ["delete", "reset", "wipe", "clear", "danger", "live trade",
     "panic", "shutdown", "factory reset"]

If you truly want to click literally everything, remove or edit SKIP_TEXT below.

Requirements
------------
From repo root:

    pip install playwright
    playwright install chromium

Run:

    python scripts/run_ui_button_double_test.py
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import http.client
from playwright.async_api import async_playwright, Page


ROOT = Path(__file__).resolve().parents[1]
OUTDIR_ROOT = ROOT / "logs" / "ui_buttons_double"
OUTDIR_ROOT.mkdir(parents=True, exist_ok=True)

TS = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
OUTDIR = OUTDIR_ROOT / TS
OUTDIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTDIR / "button_double_test_log.txt"
SUMMARY_JSON = OUTDIR / "button_double_test_summary.json"
DETAILS_JSON = OUTDIR / "button_double_test_details.json"
META_JSON = OUTDIR / "button_double_test_meta.json"

APP_URL = "http://localhost:5000"

FRONTEND_WAIT_TIMEOUT = 120
CLICK_PASSES_PER_VIEW = 2
CLICK_SETTLE_MS = 600

SKIP_TEXT = [
    "delete",
    "reset",
    "wipe",
    "clear",
    "danger",
    "live trade",
    "live trading",
    "start live",
    "panic",
    "shutdown",
    "factory reset",
]


@dataclass
class ButtonClickRecord:
    view_id: str
    pass_id: int
    index_in_dom: int
    text: str
    aria_label: str
    role: str
    skipped: bool
    success: bool
    error: str


def log(msg: str) -> None:
    msg = str(msg)
    print(msg)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def check_frontend_health() -> bool:
    try:
        conn = http.client.HTTPConnection("localhost", 5000, timeout=3)
        conn.request("GET", "/")
        resp = conn.getresponse()
        conn.close()
        return resp.status < 500
    except Exception:
        return False


async def wait_for_frontend() -> None:
    log(f"[INIT] Waiting for frontend at {APP_URL} ...")
    start = time.time()
    while True:
        if check_frontend_health():
            log("[INIT] Frontend is reachable.")
            return
        if time.time() - start > FRONTEND_WAIT_TIMEOUT:
            raise RuntimeError(
                f"Frontend not reachable after {FRONTEND_WAIT_TIMEOUT} seconds. "
                f"Start the UI (npm run dev or equivalent) and try again."
            )
        time.sleep(2)


def should_skip_button_label(label: str) -> bool:
    t = (label or "").strip().lower()
    if not t:
        return False
    return any(bad in t for bad in SKIP_TEXT)


async def double_test_buttons_on_view(page: Page, view_id: str) -> List[ButtonClickRecord]:
    """
    On the CURRENT DOM/view:
      - find all button-like elements
      - click each button in multiple passes
    Returns a list of ButtonClickRecord entries.
    """
    records: List[ButtonClickRecord] = []

    for pass_id in range(1, CLICK_PASSES_PER_VIEW + 1):
        log(f"[VIEW {view_id}] PASS {pass_id} — scanning for buttons...")
        locator = page.locator("button, [role='button'], [data-testid='button']")
        count = await locator.count()
        log(f"[VIEW {view_id}] PASS {pass_id} — found {count} button-like elements.")

        for i in range(count):
            btn = locator.nth(i)

            try:
                text = (await btn.inner_text()).strip()
            except Exception:
                text = ""

            aria = (await btn.get_attribute("aria-label")) or ""
            role = (await btn.get_attribute("role")) or ""

            label = text or aria or f"button_{i}"

            if should_skip_button_label(label):
                log(f"[VIEW {view_id}] PASS {pass_id} — SKIP destructive button #{i}: '{label}'")
                records.append(
                    ButtonClickRecord(
                        view_id=view_id,
                        pass_id=pass_id,
                        index_in_dom=i,
                        text=text,
                        aria_label=aria,
                        role=role,
                        skipped=True,
                        success=True,
                        error="",
                    )
                )
                continue

            log(f"[VIEW {view_id}] PASS {pass_id} — clicking button #{i}: '{label}'")
            success = True
            err_msg = ""

            try:
                await btn.scroll_into_view_if_needed()
                await btn.click()
                await page.wait_for_timeout(CLICK_SETTLE_MS)
            except Exception as e:
                success = False
                err_msg = str(e)
                log(f"[VIEW {view_id}] PASS {pass_id} — ERROR on button #{i} '{label}': {e!r}")

            records.append(
                ButtonClickRecord(
                    view_id=view_id,
                    pass_id=pass_id,
                    index_in_dom=i,
                    text=text,
                    aria_label=aria,
                    role=role,
                    skipped=False,
                    success=success,
                    error=err_msg,
                )
            )

    return records


async def click_nav_links(page: Page) -> List[str]:
    """
    Find nav/aside/header links and click each once.
    Returns a list of "view IDs" derived from link labels we visited.
    """
    visited_view_ids: List[str] = []

    nav_selector = "nav a, aside a, header a, a[role='menuitem']"
    nav = page.locator(nav_selector)
    count = await nav.count()
    log(f"[NAV] Found {count} navigation links.")

    for i in range(count):
        link = nav.nth(i)

        try:
            text = (await link.inner_text()).strip()
        except Exception:
            text = ""

        href = await link.get_attribute("href")
        label = text or href or f"nav_{i}"
        view_id = f"nav_{i}_{label.replace(' ', '_')[:40]}"

        log(f"[NAV] Clicking navigation link #{i}: '{label}'")
        try:
            await link.scroll_into_view_if_needed()
            await link.click()
            await page.wait_for_timeout(1000)
            visited_view_ids.append(view_id)
        except Exception as e:
            log(f"[NAV] ERROR clicking link #{i} '{label}': {e!r}")

    return visited_view_ids


async def exercise_core_labels(page: Page) -> List[str]:
    """
    Try to click through important core views by text label.
    Returns additional view IDs.
    """
    core_labels = [
        "Dashboard",
        "Scanner",
        "Universal Scanner",
        "Protocols",
        "Tier Protocols",
        "Learning Protocols",
        "MonsterRunner",
        "ApexCore",
        "ApexLab",
        "Test Lab",
        "Settings",
        "Config",
        "Configuration",
        "About",
        "Help",
    ]

    view_ids = []

    for label in core_labels:
        locator = page.get_by_text(label, exact=False)
        try:
            count = await locator.count()
            if count == 0:
                continue
            elem = locator.nth(0)
            log(f"[CORE] Attempting to open core view '{label}'")
            await elem.scroll_into_view_if_needed()
            await elem.click()
            await page.wait_for_timeout(900)
            vid = f"core_{label.replace(' ', '_')[:40]}"
            view_ids.append(vid)
        except Exception as e:
            log(f"[CORE] Could not open '{label}': {e!r}")

    return view_ids


async def main_async() -> None:
    meta: Dict[str, Any] = {
        "timestamp_utc": TS,
        "app_url": APP_URL,
        "outdir": str(OUTDIR.relative_to(ROOT)),
        "click_passes_per_view": CLICK_PASSES_PER_VIEW,
        "click_settle_ms": CLICK_SETTLE_MS,
        "skip_text": SKIP_TEXT,
    }
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log("===================================================================")
    log("QuantraCore Apex — UI Button Double-Test Harness")
    log("===================================================================")
    log(f"[INIT] Output directory: {OUTDIR}")
    log("")

    await wait_for_frontend()

    all_records: List[ButtonClickRecord] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=150)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()

        log(f"[INIT] Opening {APP_URL}")
        await page.goto(APP_URL, wait_until="networkidle")

        log("[STEP] Double-testing buttons on initial view...")
        initial_records = await double_test_buttons_on_view(page, view_id="initial_view")
        all_records.extend(initial_records)

        log("[STEP] Walking core labeled views...")
        core_view_ids = await exercise_core_labels(page)
        for view_id in core_view_ids:
            log(f"[STEP] Double-testing buttons on view '{view_id}'...")
            recs = await double_test_buttons_on_view(page, view_id=view_id)
            all_records.extend(recs)

        log("[STEP] Walking nav links...")
        nav_view_ids = await click_nav_links(page)
        for view_id in nav_view_ids:
            log(f"[STEP] Double-testing buttons on view '{view_id}'...")
            recs = await double_test_buttons_on_view(page, view_id=view_id)
            all_records.extend(recs)

        log("")
        log("[DONE] Button double-test complete. Browser will remain open.")
        log("       Close the browser window when you are finished reviewing.")
        META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        success_records = [r for r in all_records if r.success or r.skipped]
        fail_records = [r for r in all_records if (not r.success) and (not r.skipped)]

        summary = {
            "timestamp_utc": TS,
            "total_buttons_attempted": len(all_records),
            "successful_or_skipped": len(success_records),
            "failures": len(fail_records),
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        DETAILS_JSON.write_text(
            json.dumps([asdict(r) for r in all_records], indent=2),
            encoding="utf-8",
        )

        while True:
            await asyncio.sleep(60)


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        log("[INTERRUPT] Stopped by user.")
    except Exception as e:
        log(f"[FATAL] {e!r}")
        raise


if __name__ == "__main__":
    main()
