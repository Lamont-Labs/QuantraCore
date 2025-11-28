#!/usr/bin/env python3
"""
QuantraCore Apex — ApexDesk Live UI Demo & Click-Through Tester
Author: Lamont Labs (Jesse J. Lamont)

Purpose
-------
Launch a REAL browser (headed) so you can WATCH the ApexDesk dashboard being
tested live in Replit. The script:

  • Waits for the ApexDesk frontend on http://localhost:5000
  • Opens Chromium in headed mode (no headless)
  • Walks the main dashboard, clicking through:
      - All navigation links
      - All visible buttons (with safety filters)
      - Key form controls (dropdowns, toggles)
  • Takes screenshots as it moves through the UI
  • Logs everything to stdout and to logs/ui_live_demo/<timestamp>/

Safety:
  • Skips destructive buttons with text like:
        ["delete", "reset", "wipe", "clear", "danger", "live trade"]
  • You can add more skip phrases to the SKIP_TEXT list below.

Requirements
------------
From repo root in Replit:

    pip install playwright
    playwright install chromium

Then run:

    python scripts/run_ui_live_demo.py

You should see a browser window open inside Replit and watch the script
exercise the dashboard live.

This script does NOT modify any code. It only drives the browser.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List

import http.client

from playwright.async_api import async_playwright, Page


ROOT = Path(__file__).resolve().parents[1]
OUTDIR_ROOT = ROOT / "logs" / "ui_live_demo"
OUTDIR_ROOT.mkdir(parents=True, exist_ok=True)

TS = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
OUTDIR = OUTDIR_ROOT / TS
OUTDIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTDIR / "ui_live_demo_log.txt"
RUN_META = OUTDIR / "ui_live_demo_meta.json"

APEXDESK_URL = "http://localhost:5000"

SKIP_TEXT = [
    "delete",
    "reset",
    "wipe",
    "clear",
    "danger",
    "live trade",
    "start live",
    "panic",
    "shutdown",
    "factory reset",
]

FRONTEND_WAIT_TIMEOUT = 120


def log(msg: str) -> None:
    msg = str(msg)
    print(msg)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def check_frontend_health(url: str) -> bool:
    """
    Very simple HTTP GET check to see if the frontend responds with any status.
    """
    try:
        conn = http.client.HTTPConnection("localhost", 5000, timeout=3)
        conn.request("GET", "/")
        resp = conn.getresponse()
        conn.close()
        return resp.status < 500
    except Exception:
        return False


async def wait_for_frontend() -> None:
    """
    Wait for ApexDesk frontend to be reachable on localhost:5000.
    """
    log("Waiting for ApexDesk frontend on http://localhost:5000 ...")
    start = time.time()
    while True:
        if check_frontend_health(APEXDESK_URL):
            log("Frontend reachable. Continuing.")
            return
        if time.time() - start > FRONTEND_WAIT_TIMEOUT:
            raise RuntimeError(
                f"Frontend not reachable after {FRONTEND_WAIT_TIMEOUT} seconds. "
                f"Start ApexDesk (npm run dev or equivalent) and try again."
            )
        time.sleep(2)


async def screenshot(page: Page, label: str) -> None:
    safe_label = label.replace(" ", "_").replace("/", "_")
    path = OUTDIR / f"{safe_label}.png"
    await page.screenshot(path=path)
    log(f"[SCREENSHOT] {label} → {path.relative_to(ROOT)}")


async def click_all_nav_links(page: Page) -> None:
    """
    Click through navigation links once: sidebar, top nav, etc.
    """
    log("Scanning for navigation links...")
    nav_locators = page.locator("nav a, aside a, header a, a[role='menuitem']")
    count = await nav_locators.count()
    log(f"Found {count} nav links.")
    for i in range(count):
        link = nav_locators.nth(i)
        text = (await link.inner_text()).strip()
        href = await link.get_attribute("href")
        if not text and not href:
            continue

        label = text or href or f"link_{i}"
        log(f"[NAV] Clicking nav link #{i}: '{label}'")
        try:
            await link.scroll_into_view_if_needed()
            await screenshot(page, f"nav_before_{i}_{label}")
            await link.click()
            await page.wait_for_timeout(800)
            await screenshot(page, f"nav_after_{i}_{label}")
        except Exception as e:
            log(f"[NAV] Error clicking link #{i} '{label}': {e}")


async def should_skip_button_text(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return False
    for bad in SKIP_TEXT:
        if bad in t:
            return True
    return False


async def click_all_buttons(page: Page, pass_id: int) -> None:
    """
    Click as many non-destructive buttons as we can find on the current view.
    Runs once per pass. We re-scan DOM each time.
    """
    log(f"[PASS {pass_id}] Scanning for buttons...")
    locator = page.locator("button, [role='button'], [data-testid='button']")
    count = await locator.count()
    log(f"[PASS {pass_id}] Found {count} button-like elements.")

    for i in range(count):
        btn = locator.nth(i)
        try:
            text = (await btn.inner_text()).strip()
        except Exception:
            text = ""

        aria = await btn.get_attribute("aria-label")
        label = text or aria or f"button_{i}"

        if await should_skip_button_text(label):
            log(f"[PASS {pass_id}] [SKIP] Destructive button #{i}: '{label}'")
            continue

        log(f"[PASS {pass_id}] Clicking button #{i}: '{label}'")
        try:
            await btn.scroll_into_view_if_needed()
            await screenshot(page, f"btn_before_p{pass_id}_{i}_{label}")
            await btn.click()
            await page.wait_for_timeout(700)
            await screenshot(page, f"btn_after_p{pass_id}_{i}_{label}")
        except Exception as e:
            log(f"[PASS {pass_id}] Error clicking button #{i} '{label}': {e}")


async def exercise_core_views(page: Page) -> None:
    """
    Try to hit the most important views by looking for obvious labels.
    Adjust the labels here if your UI uses different wording.
    """
    key_views = [
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
        "About",
    ]

    for label in key_views:
        log(f"[CORE VIEW] Trying to open '{label}'")
        locator = page.get_by_text(label, exact=False)
        try:
            count = await locator.count()
            if count == 0:
                continue
            elem = locator.nth(0)
            await elem.scroll_into_view_if_needed()
            await screenshot(page, f"core_before_{label}")
            await elem.click()
            await page.wait_for_timeout(900)
            await screenshot(page, f"core_after_{label}")
        except Exception as e:
            log(f"[CORE VIEW] Could not open '{label}': {e}")


async def run_live_demo() -> None:
    meta = {
        "timestamp_utc": TS,
        "apexdesk_url": APEXDESK_URL,
        "outdir": str(OUTDIR.relative_to(ROOT)),
    }

    log("=================================================================")
    log("QuantraCore Apex — ApexDesk Live UI Demo & Click-Through Tester")
    log("=================================================================")
    log(f"Output directory: {OUTDIR}")
    log("")

    await wait_for_frontend()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=200)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()

        log(f"Opening ApexDesk at {APEXDESK_URL}")
        await page.goto(APEXDESK_URL, wait_until="networkidle")
        await screenshot(page, "initial_dashboard")

        await exercise_core_views(page)
        await click_all_nav_links(page)

        for pass_id in range(1, 4):
            log(f"==================== BUTTON PASS {pass_id} ====================")
            await click_all_buttons(page, pass_id=pass_id)

        log("")
        log("Live demo click-through complete.")
        log("The browser will remain open. Close the window when you're done.")
        meta["status"] = "completed"

        RUN_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        while True:
            await asyncio.sleep(60)


def main() -> None:
    try:
        asyncio.run(run_live_demo())
    except KeyboardInterrupt:
        log("Interrupted by user.")
    except Exception as e:
        log(f"[FATAL] {e}")
        raise


if __name__ == "__main__":
    main()
