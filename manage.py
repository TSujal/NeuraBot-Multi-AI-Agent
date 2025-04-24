import pandas as pd
import glob
import json
import logging
import subprocess
import time
import os
import re
from uuid import uuid4
from groq import Groq
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("manage.log")
    ]
)

# Ensure UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Initialize Groq client
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    logging.error(f"Failed to initialize Groq client: {e}")
    raise

def read_file(file_path):
    """Read the content from the given file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        raise

def write_json(file_path, data):
    """Write the JSON file for action_playwright.py."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Successfully wrote JSON to {file_path}")
    except Exception as e:
        logging.error(f"Error writing to {file_path}: {e}")
        raise

def validate_json(data):
    """Validate JSON structure."""
    try:
        json.dumps(data)
        return True
    except Exception:
        return False

def extract_json_from_response(response_text):
    """Extract JSON block from a response containing markdown or text."""
    try:
        # Look for JSON within ```json ``` or standalone
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, response_text)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)
        
        # Try parsing the entire response as JSON
        return json.loads(response_text.strip())
    except json.JSONDecodeError as e:
        # Fallback: Remove any <think> tags and try extracting JSON again
        cleaned_response = re.sub(r'<think>[\s\S]*?</think>', '', response_text).strip()
        json_match = re.search(r'\[[\s\S]*\]', cleaned_response)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        logging.error(f"Failed to extract JSON from response: {e}")
        raise ValueError(f"Invalid JSON in response: {response_text}")

def extract_domain(url):
    """Extract the primary domain name from a URL using regex."""
    try:
        pattern = r'^(?:https?:\/\/)?(?:[^@\n]+@)?(?:www\.)?([^:\/\n?]+)'
        match = re.match(pattern, url)
        if not match:
            raise ValueError(f"Cannot extract domain from URL: {url}")
        domain = match.group(1).split('.')[-2]
        logging.info(f"Extracted domain: {domain}")
        return domain
    except Exception as e:
        logging.error(f"Failed to extract domain from URL {url}: {e}")
        raise

def find_steps_file(task_timestamp=None):
    """Locate the .txt file in the steps/ directory matching the task timestamp."""
    try:
        steps_dir = os.path.join(os.path.dirname(__file__), "steps")
        txt_files = glob.glob(os.path.join(steps_dir, "*.txt"))
        if not txt_files:
            raise FileNotFoundError("No .txt files found in steps/ directory")
        
        if task_timestamp:
            matching_files = [f for f in txt_files if task_timestamp in os.path.basename(f)]
            if not matching_files:
                raise FileNotFoundError(f"No steps file found for timestamp {task_timestamp}")
            if len(matching_files) > 1:
                logging.warning(f"Multiple .txt files found for timestamp {task_timestamp}: {matching_files}. Using the first one.")
            steps_file = matching_files[0]
        else:
            txt_files.sort(key=os.path.getmtime, reverse=True)
            steps_file = txt_files[0]
        
        logging.info(f"Found steps file: {steps_file}")
        return steps_file
    except Exception as e:
        logging.error(f"Failed to find steps file: {e}")
        raise

def check_widgets_file(domain):
    """Check if the {domain}_widgets.txt file exists and return its path."""
    widgets_txt_file = os.path.join(os.path.dirname(__file__), "Web_Scrap", "scraped_data", f"{domain}_widgets.txt")
    if not os.path.exists(widgets_txt_file):
        logging.error(f"Widgets file not found: {widgets_txt_file}")
        return None
    logging.info(f"Widgets file found: {widgets_txt_file}")
    return widgets_txt_file

def select_widget_task_json(domain, task_timestamp):
    """
    Generate a task.json file using LLM based on steps/**.txt and Web_Scrap/scraped_data/<domain>_widgets.txt.
    The task.json will navigate to the website, optionally perform a search if specified, apply filters, and log the final URL.
    Save the task.json file in the Task folder.
    """
    try:
        widgets_txt_file = check_widgets_file(domain)
        if not widgets_txt_file:
            raise FileNotFoundError(f"Widgets file for {domain} not found")

        plan_file = find_steps_file(task_timestamp)
        task_json_path = os.path.join(os.path.dirname(__file__), "Task", "task.json")

        widgets_content = read_file(widgets_txt_file)
        steps_content = read_file(plan_file)

        # Extract the base URL from the steps file
        url_match = re.search(r'https?://[^\s]+', steps_content)
        if not url_match:
            raise ValueError("No URL found in steps file")
        base_url = url_match.group(0)

        # Check if a search term is specified in the steps file (e.g., "search for lenovo laptop")
        search_term = None
        search_match = re.search(r'search\s+(?:for\s+)?["\']?([^"\']+)["\']?', steps_content, re.IGNORECASE)
        if search_match:
            search_term = search_match.group(1).strip()
            logging.info(f"Search term found: {search_term}")

        system_prompt = """
You are an AI assistant tasked with creating a task.json file to apply website filters based on user requirements and available filter options.
Input:
- widgets_txt_file.txt: Contains available filter options (e.g., brand, size, price range) with their CSS selectors.
- plan_file.txt: User requirements (e.g., specific product features, budget, delivery preferences) and task description.
- base_url: The starting URL of the website.
- search_term: Optional search query to enter in the website's search bar (may be null).

Output:
A JSON list of tasks to:
1. Navigate to the base_url.
2. If search_term is provided, enter the search term in the search bar and submit it.
3. Apply all relevant filters from widgets.txt that match the user requirements in plan_file.txt.
4. Wait for page reloads after each filter application.
5. Log the final URL after all filters are applied.

Each task must include:
- action: The action to perform (e.g., "goto", "click", "type", "wait_for_timeout", "log").
- url: For "goto" action (e.g., the base URL).
- selector: CSS selector for filter elements or search bar (from widgets.txt or predefined for search).
- timeout: Timeout in milliseconds for waiting actions (e.g., 2000 for reloads, 1000 for delays).
- text: For "type" action (e.g., search term).
- message: For "log" action to record actions or final URL.

Rules:
- Start with a "goto" action to the base_url.
- If search_term is provided, include:
  - "click" on the search bar (selector: "#twotabsearchtextbox").
  - "type" to enter the search_term.
  - "press" to submit the search (key: "Enter").
  - "wait_for_timeout" (2000ms) for results to load.
  - "log" to record the search action.
- For each filter, include:
  - "scroll_into_view" to ensure the filter is visible.
  - "hover" to simulate user interaction.
  - "click" to apply the filter.
  - "wait_for_timeout" (2000ms) to allow page reload.
  - "log" to record the filter applied.
- Include a 1-second delay between actions (via wait_for_timeout: 1000).
- After all filters, include:
  - "wait_for_selector" to ensure search results load (selector: "div[data-component-type='s-search-result']", timeout: 1500).
  - "log" to capture the final URL with "{{page.url}}".
- Only include filters that align with the user requirements in plan_file.txt.
- If no matching filters are found, include only the "goto", optional search, and "log" actions.
- Return ONLY the JSON list, without any explanatory text, markdown, or <think> tags. Do not include ```json or other wrappers.

Example for reference (do not copy strictly):
[
    {"action": "goto", "url": "https://www.amazon.com"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "click", "selector": "#twotabsearchtextbox"},
    {"action": "type", "selector": "#twotabsearchtextbox", "text": "lenovo laptop"},
    {"action": "press", "selector": "#twotabsearchtextbox", "key": "Enter"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "log", "message": "Searched for lenovo laptop"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "scroll_into_view", "selector": "span.a-size-base:text('Lenovo')"},
    {"action": "hover", "selector": "span.a-size-base:text('Lenovo')"},
    {"action": "click", "selector": "span.a-size-base:text('Lenovo')"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "log", "message": "Applied filter: Lenovo"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "wait_for_selector", "selector": "div[data-component-type='s-search-result']", "timeout": 1500},
    {"action": "log", "message": "Final URL: {{page.url}}"}
]
"""

        user_prompt = f"""
Base URL: {base_url}

Widgets:
{widgets_content}

User Requirements and Steps:
{steps_content}

Search Term: {search_term if search_term else 'null'}
"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="Deepseek-R1-Distill-Llama-70b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=5000,
                    temperature=0.5
                )

                raw_response = response.choices[0].message.content
                logging.info(f"Raw LLM response: {raw_response}")
                
                if not raw_response.strip():
                    raise ValueError("Empty response from LLM")

                task_json = extract_json_from_response(raw_response)

                if not isinstance(task_json, list) or not all(isinstance(task, dict) for task in task_json):
                    raise ValueError("Invalid task.json format: Expected a list of task dictionaries")

                write_json(task_json_path, task_json)
                logging.info(f"Generated task.json at {task_json_path}")
                return task_json_path
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to parse LLM response after {max_retries} attempts: {raw_response}")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(5)

    except Exception as e:
        logging.error(f"Failed to generate first task.json: {e}")
        raise

def run_action_playwright():
    """
    Run action_playwright.py using the task.json file in the Task folder.
    Return the output URL and domain.
    """
    try:
        task_json_path = os.path.join(os.path.dirname(__file__), "Task", "task.json")
        if not os.path.exists(task_json_path):
            raise FileNotFoundError(f"task.json not found at {task_json_path}")
        
        result = subprocess.run(
            ["python", os.path.join("Automate_action", "action_playwright.py")],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )

        if result.returncode != 0:
            raise RuntimeError(f"action_playwright.py failed: {result.stderr}")

        # Extract the final URL from stdout
        stdout_lines = result.stdout.splitlines()
        url = None
        for line in stdout_lines:
            if line.startswith("Final URL:"):
                url = line.replace("Final URL:", "").strip()
                break
        if not url or not url.startswith("http"):
            raise ValueError(f"Invalid or missing URL in action_playwright.py output: {result.stdout}")

        domain = extract_domain(url)
        logging.info(f"action_playwright.py output URL: {url}, domain: {domain}")
        return url, domain
    except Exception as e:
        logging.error(f"Failed to run action_playwright.py: {e}")
        raise

def find_csv_file():
    """
    Locate the CSV file generated by product.py in the Web_Scrap/scraped_data directory.
    """
    try:
        output_dir = os.path.join(os.path.dirname(__file__), "Web_Scrap", "scraped_data")
        csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in Web_Scrap/scraped_data directory")
        csv_files.sort(key=os.path.getmtime, reverse=True)
        csv_file = csv_files[0]
        logging.info(f"Found CSV file: {csv_file}")
        return csv_file
    except Exception as e:
        logging.error(f"Failed to find CSV file: {e}")
        raise

def run_product_scraper(url, domain):
    """
    Run product.py with the URL and domain to fetch top 10 products and save to CSV.
    """
    try:
        result = subprocess.run(
            ["python", os.path.join("Web_Scrap", "product.py"), url, domain],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        if result.returncode != 0:
            raise RuntimeError(f"product.py failed: {result.stderr}")

        csv_file = find_csv_file()
        logging.info(f"product.py completed, found generated CSV")
        return csv_file
    except Exception as e:
        logging.error(f"Failed to run product.py: {e}")
        raise

def select_product_and_generate_task_json(csv_file):
    """
    Use LLM to select a product from the CSV that matches the description in plan_file,
    then generate a task.json to navigate to the product's URL, add it to cart, and take a screenshot.
    """
    try:
        task_json_path = os.path.join(os.path.dirname(__file__), "Task", "task.json")
        df = pd.read_csv(csv_file)
        if not all(col in df.columns for col in ['Title', 'Price', 'Rating', 'Reviews', 'URL']):
            raise ValueError("CSV missing required columns: [Title, Price, Rating, Reviews, URL]")
        products = df.to_dict('records')

        plan_file = find_steps_file()
        steps_content = read_file(plan_file)

        system_prompt = """
You are an AI assistant tasked with selecting a product from a CSV file that best matches the user's requirements and creating a task.json to navigate to its URL, add it to the cart, and take a screenshot.
Input:
- CSV: Contains [Title, Price, Rating, Reviews, URL] for top 10 products.
- plan_file.txt: Description of the desired product, including specifications, budget, and preferences.

Output:
A JSON list of tasks to:
1. Navigate to the selected product's URL.
2. Click the "Add to Cart" button.
3. Handle any warranty or pop-up by clicking "No Thanks" or similar.
4. Take a screenshot of the cart confirmation.
5. Log the final URL.

Each task must include:
- action: The action to perform (e.g., "goto", "click", "screenshot", "log").
- url: For "goto" action (the product URL).
- selector: CSS selector for buttons (e.g., "#add-to-cart-button", "button:has-text('No Thanks')").
- path: File path for screenshot (e.g., "cart_confirmation.png").
- timeout: Timeout in milliseconds for waiting actions (e.g., 2000).
- message: For "log" action to record the final URL.

Rules:
- Select the product from the CSV that best matches the requirements in plan_file.txt (e.g., title, price within budget, rating).
- Use a generic add-to-cart selector (e.g., "#add-to-cart-button, [value='Add to Cart']") to ensure compatibility.
- Include a step to handle warranty/pop-up with a selector like "button:has-text('No Thanks'), button:has-text('Decline')".
- Use a 1-second delay between actions (via wait_for_timeout: 1000).
- Save the screenshot as "cart_confirmation.png".
- Return ONLY the JSON list, without any explanatory text, markdown, or <think> tags. Do not include ```json or other wrappers.

Example for reference (do not copy strictly):
[
    {"action": "goto", "url": "https://www.amazon.com/dp/B08N5WRWNW"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "scroll_into_view", "selector": "#add-to-cart-button"},
    {"action": "hover", "selector": "#add-to-cart-button"},
    {"action": "click", "selector": "#add-to-cart-button"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "click", "selector": "button:has-text('No Thanks'), button:has-text('Decline')"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "screenshot", "path": "cart_confirmation.png"},
    {"action": "log", "message": "Final URL: {{page.url}}"}
]
"""

        user_prompt = f"""
Products:
{json.dumps(products, indent=2)}

User Requirements and Steps:
{steps_content}
"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="Deepseek-R1-Distill-Llama-70b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=5000,
                    temperature=0.5
                )

                raw_response = response.choices[0].message.content
                logging.info(f"Raw LLM response for product selection: {raw_response}")
                task_json = extract_json_from_response(raw_response)

                if not validate_json(task_json) or not isinstance(task_json, list):
                    raise ValueError("Invalid task.json format: Expected a list of tasks")

                write_json(task_json_path, task_json)
                logging.info(f"Generated task.json for product selection and cart addition at {task_json_path}")
                return task_json_path
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to parse LLM response after {max_retries} attempts: {raw_response}")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(5)

    except Exception as e:
        logging.error(f"Failed to generate second task.json: {e}")
        raise

def run_action_playwright_for_cart():
    """
    Run action_playwright.py to execute the second task.json (add product to cart and take screenshot).
    """
    try:
        task_json_path = os.path.join(os.path.dirname(__file__), "Task", "task.json")
        if not os.path.exists(task_json_path):
            raise FileNotFoundError(f"task.json not found at {task_json_path}")
        
        result = subprocess.run(
            ["python", os.path.join("Automate_action", "action_playwright.py")],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )

        if result.returncode != 0:
            raise RuntimeError(f"action_playwright.py for cart failed: {result.stderr}")

        output = result.stdout.strip()
        logging.info(f"action_playwright.py for cart completed: {output}")
        print(f"Chosen product added to cart: {output}")
    except Exception as e:
        logging.error(f"Failed to run action_playwright.py for cart: {e}")
        raise

def main(task_timestamp=None):
    """Main function to orchestrate the workflow."""
    try:
        plan_file = find_steps_file(task_timestamp)
        steps_content = read_file(plan_file)
        url_match = re.search(r'https?://[^\s]+', steps_content)
        if not url_match:
            raise ValueError("No URL found in steps file")
        url = url_match.group(0)
        domain = extract_domain(url)
        logging.info(f"Starting workflow with domain: {domain}")

        if not check_widgets_file(domain):
            raise FileNotFoundError(f"Widgets file for {domain} not found. Ensure {domain}_widgets.txt exists in Web_Scrap/scraped_data/")

        select_widget_task_json(domain, task_timestamp)
        url, domain = run_action_playwright()
        time.sleep(1)
        csv_file = run_product_scraper(url, domain)
        select_product_and_generate_task_json(csv_file)
        run_action_playwright_for_cart()

        logging.info("Workflow completed successfully")
        return {"status": "success", "message": "Workflow completed"}
    except Exception as e:
        logging.error(f"Workflow failed: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    task_timestamp = sys.argv[1] if len(sys.argv) > 1 else None
    result = main(task_timestamp)
    print(json.dumps(result, ensure_ascii=False))