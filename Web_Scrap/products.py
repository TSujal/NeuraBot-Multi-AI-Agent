from playwright.sync_api import sync_playwright
import pandas as pd
import time
from urllib.parse import urlparse, parse_qs
import os
import json
import re

def load_site_config(domain):
    """Load site-specific CSS selectors from site_config_product/ directory in parent directory."""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "site_config_product")
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

def extract_widgets_and_options(url, css_selector=""):
    widget_data = {}

    # Validate css_selector
    if css_selector and ("/" in css_selector or "\\" in css_selector):
        print(f"Warning: '{css_selector}' looks like a file path, ignoring it.")
        css_selector = ""

    # Parse domain from URL
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace("www.", "")
    # Extract product name from query (e.g., k=laptop or q=shoes)
    query_params = parse_qs(parsed_url.query)
    product_name = query_params.get('k', query_params.get('q', ['products']))[0]
    # Sanitize product_name for filename
    product_name = re.sub(r'[^\w\s-]', '', product_name).strip().replace(' ', '_').lower()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Visible browser
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
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
            return widget_data

        config = load_site_config(domain)

        # Try the provided selector first, if any
        if css_selector:
            print(f"\nTrying provided selector: '{css_selector}'")
            widgets = page.query_selector_all(f"{css_selector}")
            print(f"Found {len(widgets)} elements.")
        else:
            widgets = []

        # If no widgets, try product selectors from config
        if not widgets:
            for selector in config["product_selectors"]:
                print(f"\nTrying selector: '{selector}'")
                widgets = page.query_selector_all(f"{selector}")
                print(f"Found {len(widgets)} elements.")
                if widgets:
                    print(f"Using '{selector}' as the selector.")
                    break
        
        if not widgets:
            print("No products found with any selector. Check the page structure or config selectors.")
            browser.close()
            return widget_data

        # Prepare CSV data
        csv_data = []

        # Process up to 10 products
        for widget in widgets[:10]:
            try:
                # Highlight the product box
                page.evaluate("""
                    (element) => {
                        element.style.border = '3px solid red';
                        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                """, widget)
                
                # Extract title
                title_elem = widget.query_selector(config["title"])
                widget_name = title_elem.text_content().strip() if title_elem else "Unknown Title"
                print(f"Processing product: '{widget_name}'")

                # Extract product details
                options = []

                # Price
                price = "N/A"
                price_whole = widget.query_selector(config["price_whole"])
                if config.get("price_fraction"):
                    price_fraction = widget.query_selector(config["price_fraction"])
                    price = (price_whole.text_content().strip() + 
                             (price_fraction.text_content().strip() if price_fraction else "")) if price_whole else "N/A"
                else:
                    price = price_whole.text_content().strip() if price_whole else "N/A"
                options.append(f"Price: {price}")

                # Rating
                rating_elem = widget.query_selector(config["rating"])
                rating = rating_elem.text_content().strip() if rating_elem else "N/A"
                options.append(f"Rating: {rating}")

                # Reviews
                reviews_elem = widget.query_selector(config["reviews"])
                reviews = reviews_elem.text_content().strip() if reviews_elem else "N/A"
                options.append(f"Reviews: {reviews}")

                # URL
                link_elem = widget.query_selector(config["url"])
                url_val = link_elem.get_attribute("href") if link_elem else "N/A"
                if url_val != "N/A" and not url_val.startswith("http") and config.get("url_prefix"):
                    url_val = f"{config['url_prefix']}{url_val}"
                options.append(f"URL: {url_val}")

                widget_data[widget_name] = options

                # Add to CSV data
                csv_data.append({
                    "Title": widget_name,
                    "Price": price,
                    "Rating": rating,
                    "Reviews": reviews,
                    "URL": url_val
                })

                # Remove highlight
                page.evaluate("""
                    (element) => {
                        element.style.border = '';
                    }
                """, widget)

                # Wait 2 seconds
                page.wait_for_timeout(2000)

            except Exception as e:
                print(f"Failed to process '{widget_name}': {e}")
                continue

        browser.close()

    # Save to CSV in scraped_data/
    if csv_data:
        df = pd.DataFrame(csv_data)
        output_dir = os.path.join(os.path.dirname(__file__), "scraped_data")
        os.makedirs(output_dir, exist_ok=True)
        csv_file = os.path.join(output_dir, f"{domain.split('.')[0]}_{product_name}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"\nData saved to {csv_file}")

    # Output results to terminal
    if not widget_data:
        print("\nNo products or details extracted.")
    else:
        print("\nResults:")
        for widget_name, options in widget_data.items():
            print(f"{widget_name}")
            for option in options:
                print(f"  - {option}")
            print()

    return widget_data

# Prompt user for URL and domain name
url = input("Please enter the URL: ").strip()
domain_name = input("Please enter the domain name (e.g., amazon): ").strip()
css_selector = f"site_config_product/{domain_name}.json"

# Run the extraction
extract_widgets_and_options(url, css_selector)

#url = "https://www.amazon.com/s?k=nike+shoes+men&i=fashion&rh=n%3A7141123011%2Cp_n_size_browse-vebin%3A1285100011%2Cp_123%3A234394%2Cp_n_feature_nineteen_browse-bin%3A24045095011%2Cp_72%3A2661618011&dc=&crid=1D2JU1WOR902B&qid=1745261074&rnid=2661611011&sprefix=nike+%2Caps%2C173&ref=sr_nr_p_36_0_0&low-price=210&high-price="
#css_selector = "site_config_product/amazon.json"  # Empty, rely on JSON config
#extract_widgets_and_options(url, css_selector)