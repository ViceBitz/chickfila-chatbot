"""
Scrape per-serving calorie data from individual Chick-fil-A menu item pages.
Patches the calories field into data/chickfila.json.

Usage:
    python data/scrape_calories.py
"""

import asyncio
import json
import re
from pathlib import Path

from playwright.async_api import async_playwright

DATA_FILE = Path(__file__).parent / "chickfila.json"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# how many pages to load at once
CONCURRENCY = 5


async def scrape_calories_for_item(context, name: str, url: str) -> int | None:
    page = await context.new_page()
    try:
        # use 'load' — much faster than 'networkidle'
        await page.goto(url, wait_until="load", timeout=20000)

        # wait up to 5s for a calorie element to appear
        cal_selectors = [
            "[class*='calorie' i]",
            "[class*='Calorie' i]",
            "[data-testid*='calorie' i]",
            "text=/\\d+ Cal/i",
        ]
        cal_text = None
        for sel in cal_selectors:
            try:
                el = page.locator(sel).first
                await el.wait_for(timeout=5000)
                cal_text = (await el.inner_text()).strip()
                if cal_text:
                    break
            except Exception:
                continue

        if not cal_text:
            # fallback: search visible text on page for calorie pattern
            content = await page.content()
            m = re.search(r'"calories"\s*:\s*"?(\d+)"?', content)
            if not m:
                m = re.search(r'(\d{2,4})\s*Cal(?:ories)?', content)
            if m:
                cal_text = m.group(1)

        if cal_text:
            m = re.search(r"(\d+)", cal_text)
            if m:
                return int(m.group(1))
    except Exception as e:
        print(f"  [error] {name}: {e}")
    finally:
        await page.close()
    return None


async def main():
    data = json.loads(DATA_FILE.read_text())
    menu = data["menu"]

    # only items with a proper menu page URL and no calories yet
    to_scrape = [
        i for i in menu
        if "chick-fil-a.com/menu/" in (i.get("url") or "")
        and not i.get("calories")
    ]
    print(f"Items to scrape: {len(to_scrape)}")

    calorie_map: dict[str, int] = {}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=USER_AGENT)

        sem = asyncio.Semaphore(CONCURRENCY)

        async def bounded(item):
            async with sem:
                cal = await scrape_calories_for_item(context, item["name"], item["url"])
                if cal:
                    calorie_map[item["name"]] = cal
                    print(f"  ✓ {item['name']}: {cal} cal")
                else:
                    print(f"  - {item['name']}: not found")

        await asyncio.gather(*[bounded(i) for i in to_scrape])
        await browser.close()

    # patch calories back into menu list
    matched = 0
    for item in menu:
        if item["name"] in calorie_map:
            item["calories"] = calorie_map[item["name"]]
            matched += 1

    DATA_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nDone — matched {matched}/{len(to_scrape)} items. Written to {DATA_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
