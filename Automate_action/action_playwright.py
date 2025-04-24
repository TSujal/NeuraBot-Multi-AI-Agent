import json
import time
import random
from pathlib import Path
import os
import sys
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Ensure UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # For older Python versions
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Utility: Add natural delays between actions
def delay(seconds=1):
    time.sleep(seconds)

# Utility: Simulate human-like mouse movement
def simulate_mouse_movement(page, selector):
    try:
        element = page.query_selector(selector)
        if element:
            bounding_box = element.bounding_box()
            if bounding_box:
                x = bounding_box["x"] + bounding_box["width"] / 2
                y = bounding_box["y"] + bounding_box["height"] / 2
                page.mouse.move(x + random.uniform(-10, 10), y + random.uniform(-10, 10))
                delay()
    except:
        pass  # Skip if bounding box fails

# Utility: Close pop-ups or warranty prompts
def close_popups(page):
    try:
        # Close warranty prompts or pop-ups
        page.evaluate('''() => {
            const buttons = document.querySelectorAll('button:has-text("No Thanks"), button:has-text("Decline"), button:has-text("Continue"), .a-popover button, #sw-atc-confirmation button.close');
            buttons.forEach(btn => btn.click());
        }''')
        delay(0.5)
    except:
        pass

# Main runner
def run_automation_from_json(json_path="../Task/task.json", delay_time=1):
    # Resolve json_path relative to script's parent directory
    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, json_path)
    json_path = Path(json_path).resolve()
    
    # Check if file exists
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return

    # Load steps from JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        steps = json.load(f)

    # Start Playwright session
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=500)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            viewport={"width": 1280, "height": 720},
            locale="en-US",
            accept_downloads=True
        )
        context.clear_cookies()  # Clear cookies to avoid CAPTCHA
        page = context.new_page()
        page.set_extra_http_headers({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        })

        for step in steps:
            action = step.get("action")
            selector = step.get("selector")

            try:
                print(f"Performing action: {action}")

                if action == "goto":
                    page.goto(step["url"], wait_until="domcontentloaded")
                    page.wait_for_timeout(1000)
                    close_popups(page)  # Close any initial pop-ups
                elif action == "click":
                    page.wait_for_selector(selector, timeout=10000)
                    simulate_mouse_movement(page, selector)
                    page.eval_on_selector(selector, "el => el.scrollIntoView({ behavior: 'smooth', block: 'center' })")
                    page.wait_for_timeout(500)
                    page.click(selector)
                    close_popups(page)  # Close pop-ups after clicks
                elif action == "type":
                    page.wait_for_selector(selector, timeout=10000)
                    simulate_mouse_movement(page, selector)
                    page.eval_on_selector(selector, "el => el.scrollIntoView({ behavior: 'smooth', block: 'center' })")
                    page.wait_for_timeout(500)
                    page.fill(selector, step["text"])
                elif action == "press":
                    page.wait_for_selector(selector, timeout=10000)
                    page.press(selector, step["key"])
                elif action == "wait_for_selector":
                    timeout = step.get("timeout", 10000)
                    page.wait_for_selector(selector, timeout=timeout)
                elif action == "wait_for_timeout":
                    page.wait_for_timeout(step["timeout"])
                elif action == "evaluate":
                    if selector:
                        page.eval_on_selector(selector, step["script"])
                    else:
                        page.evaluate(step["script"])
                elif action == "screenshot":
                    page.screenshot(path=step["path"])
                elif action == "hover":
                    page.wait_for_selector(selector, timeout=10000)
                    simulate_mouse_movement(page, selector)
                    page.eval_on_selector(selector, "el => el.scrollIntoView({ behavior: 'smooth', block: 'center' })")
                    page.wait_for_timeout(500)
                    page.hover(selector)
                elif action == "scroll_into_view":
                    page.wait_for_selector(selector, timeout=10000)
                    page.eval_on_selector(selector, "el => el.scrollIntoView({ behavior: 'smooth', block: 'center' })")
                elif action == "get_text":
                    page.wait_for_selector(selector, timeout=10000)
                    text = page.inner_text(selector)
                    with open(step["save_as"], "w", encoding="utf-8") as f:
                        f.write(text)
                elif action == "assert_text":
                    page.wait_for_selector(selector, timeout=10000)
                    actual = page.inner_text(selector).strip()
                    expected = step["expected"].strip()
                    assert expected in actual, f"Assertion failed: '{expected}' not in '{actual}'"
                elif action == "focus":
                    page.wait_for_selector(selector, timeout=10000)
                    simulate_mouse_movement(page, selector)
                    page.focus(selector)
                elif action == "blur":
                    page.wait_for_selector(selector, timeout=10000)
                    page.eval_on_selector(selector, "el => el.blur()")
                elif action == "log":
                    message = step["message"].replace("{{page.url}}", page.url)
                    print(f"Log: {message}")
                else:
                    print(f"Warning: Unknown action: {action}")
                
                delay(delay_time)

            except PlaywrightTimeoutError:
                print(f"Timeout while waiting for selector: {selector}")
                if action in ["click", "hover", "scroll_into_view"] and "add-to-cart" in selector:
                    # Fallback for cart button
                    page.evaluate("document.querySelector('#add-to-cart-button, #submit.add-to-cart')?.click()")
                    close_popups(page)
                elif action == "click" and "s-search-result" in selector:
                    # Fallback for product selection
                    page.evaluate("document.querySelector('div[data-component-type=\"s-search-result\"] h2 a')?.click()")
            except Exception as e:
                print(f"Error during step '{action}': {str(e)}")
                close_popups(page)

        final_url = page.url
        print(f"\nFinal URL: {final_url}\n")
        
        print("Automation complete!")
        browser.close()

if __name__ == "__main__":
    run_automation_from_json()