'''
from playwright.sync_api import sync_playwright
import os
import json
from urllib.parse import urlparse

def load_widget_config(domain):
    """Load widget-specific CSS selectors from site_config_widget/ directory."""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "site_config_widget")
    # Use domain without .com for file name
    config_file = os.path.join(config_dir, f"{domain.split('.')[0]}.json")
    try:
        print(f"Current directory: {os.getcwd()}")
        print(f"Attempting to load config: {config_file}")
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"Loaded config for {domain}: {config}")
                return config
        else:
            raise FileNotFoundError(f"Config file not found: {config_file}. Please create a JSON config for {domain}.")
    except Exception as e:
        raise Exception(f"Error loading config for {domain}: {e}")

def extract_widget_names(url, css_selector=""):
    widget_names = []

    # Validate css_selector
    if css_selector and ("/" in css_selector or "\\" in css_selector):
        print(f"Warning: '{css_selector}' looks like a file path, ignoring it.")
        css_selector = ""

    # Parse domain from URL
    domain = urlparse(url).netloc.replace("www.", "")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Visible for debugging
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        })

        # Load the page
        print("Loading page...")
        try:
            page.goto(url, timeout=60000)
            page.wait_for_load_state("domcontentloaded", timeout=60000)
            print("Page loaded.")
        except Exception as e:
            print(f"Failed to load page: {e}")
            browser.close()
            return widget_names

        # Load widget config
        config = load_widget_config(domain)

        # Try the provided selector first, if any
        if css_selector:
            print(f"\nTrying provided selector: '{css_selector}'")
            widgets = page.query_selector_all(f"{css_selector}")
            print(f"Found {len(widgets)} potential widgets.")
        else:
            widgets = []

        # If no widgets, try selectors from config
        if not widgets:
            for selector in config["widget_selectors"]:
                print(f"\nTrying selector: '{selector}'")
                widgets = page.query_selector_all(f"{selector}")
                print(f"Found {len(widgets)} potential widgets.")
                if widgets:
                    print(f"Using '{selector}' as the selector.")
                    break

        if not widgets:
            print("No widgets found. Check the page structure or config selectors.")
            browser.close()
            return widget_names

        # Extract widget names
        for widget in widgets:
            widget_name = widget.text_content().strip()
            if widget_name and len(widget_name) >= 2:
                widget_names.append(widget_name)

        browser.close()

        # Output widget names to terminal
        if not widget_names:
            print("\nNo widget names extracted.")
        else:
            print("\nWidget Names:")
            for name in widget_names:
                print(f"- {name}")

        # Save to txt in scraped_data/
        if widget_names:
            output_dir = os.path.join(os.path.dirname(__file__), "scraped_data")
            os.makedirs(output_dir, exist_ok=True)
            txt_file = os.path.join(output_dir, f"{domain.split('.')[0]}_widgets.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                for name in widget_names:
                    f.write(f"{name}\n")
            print(f"\nWidget names saved to {txt_file}")

    return widget_names

# Example usage
url = "https://www.walmart.com/search?q=lenovo+laptop"
css_selector = "site_config_widget\walmart.json"  # site_config_widget\walmart.json
extract_widget_names(url, css_selector)
'''
from playwright.sync_api import sync_playwright
import os
import json
import sys
from urllib.parse import urlparse

def load_widget_config(domain):
    """Load widget-specific CSS selectors from site_config_widget/ directory."""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "site_config_widget")
    # Use domain without .com for file name
    config_file = os.path.join(config_dir, f"{domain.split('.')[0]}.json")
    try:
        print(f"Current directory: {os.getcwd()}")
        print(f"Attempting to load config: {config_file}")
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"Loaded config for {domain}: {config}")
                return config
        else:
            raise FileNotFoundError(f"Config file not found: {config_file}. Please create a JSON config for {domain}.")
    except Exception as e:
        raise Exception(f"Error loading config for {domain}: {e}")

def extract_widget_names(url, css_selector=""):
    widget_names = []

    # Validate css_selector
    if css_selector and ("/" in css_selector or "\\" in css_selector):
        print(f"Warning: '{css_selector}' looks like a file path, ignoring it.")
        css_selector = ""

    # Parse domain from URL
    domain = urlparse(url).netloc.replace("www.", "")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Visible for debugging
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        })

        # Load the page
        print("Loading page...")
        try:
            page.goto(url, timeout=60000)
            page.wait_for_load_state("domcontentloaded", timeout=60000)
            print("Page loaded.")
        except Exception as e:
            print(f"Failed to load page: {e}")
            browser.close()
            return widget_names

        # Load widget config
        config = load_widget_config(domain)

        # Try the provided selector first, if any
        if css_selector:
            print(f"\nTrying provided selector: '{css_selector}'")
            widgets = page.query_selector_all(f"{css_selector}")
            print(f"Found {len(widgets)} potential widgets.")
        else:
            widgets = []

        # If no widgets, try selectors from config
        if not widgets:
            for selector in config["widget_selectors"]:
                print(f"\nTrying selector: '{selector}'")
                widgets = page.query_selector_all(f"{selector}")
                print(f"Found {len(widgets)} potential widgets.")
                if widgets:
                    print(f"Using '{selector}' as the selector.")
                    break

        if not widgets:
            print("No widgets found. Check the page structure or config selectors.")
            browser.close()
            return widget_names

        # Extract widget names
        for widget in widgets:
            widget_name = widget.text_content().strip()
            if widget_name and len(widget_name) >= 2:
                widget_names.append(widget_name)

        browser.close()

        # Output widget names to terminal
        if not widget_names:
            print("\nNo widget names extracted.")
        else:
            print("\nWidget Names:")
            for name in widget_names:
                print(f"- {name}")

        # Save to txt in scraped_data/
        if widget_names:
            output_dir = os.path.join(os.path.dirname(__file__), "scraped_data")
            os.makedirs(output_dir, exist_ok=True)
            txt_file = os.path.join(output_dir, f"{domain.split('.')[0]}_widgets.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                for name in widget_names:
                    f.write(f"{name}\n")
            print(f"\nWidget names saved to {txt_file}")

    return widget_names

if __name__ == "__main__":
    # Check if input is provided via command-line argument
    if len(sys.argv) > 1:
        try:
            domain_input, product_name = sys.argv[1].split(" ", 1)
        except ValueError:
            print("Error: Please provide domain and product (e.g., 'amazon laptop')")
            sys.exit(1)
    else:
        # Prompt for input if no command-line argument
        user_input = input("Enter domain and product (e.g., amazon laptop): ").strip()
        try:
            domain_input, product_name = user_input.split(" ", 1)
        except ValueError:
            print("Error: Please provide domain and product (e.g., 'amazon laptop')")
            sys.exit(1)

    # Construct URL and CSS selector from input
    url = f"https://www.{domain_input}.com/s?k={product_name}"
    css_selector = ""  # Empty to rely on config selectors

    # Run the scraping function
    extract_widget_names(url, css_selector)

    