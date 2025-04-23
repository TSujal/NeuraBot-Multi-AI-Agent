'''
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

# Assuming action_playwright is in Automate_action directory
from Automate_action import action_playwright

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Groq client
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    logging.error(f"Failed to initialize GROQ client...TALK TO @SUJAL:{e}")
    raise

def read_file(file_path):
    """Read the content from the given file."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: Talk to @Sujal: {e}")
        raise

def write_json(file_path, data):
    """Write the JSON file for action_playwright.py."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Successfully wrote JSON to {file_path} ✅")
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
        # Look for ```json ... ``` block
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, response_text)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)
        
        # If no markdown, try parsing the response directly
        return json.loads(response_text)
    except json.JSONDecodeError as e:
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
        logging.info(f"Extracted domain: {domain} ✅")
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
            # Look for a file matching the timestamp
            matching_files = [f for f in txt_files if task_timestamp in os.path.basename(f)]
            if not matching_files:
                raise FileNotFoundError(f"No steps file found for timestamp {task_timestamp}")
            if len(matching_files) > 1:
                logging.warning(f"Multiple .txt files found for timestamp {task_timestamp}: {matching_files}. Using the first one.")
            steps_file = matching_files[0]
        else:
            # Fallback to the first file if no timestamp is provided
            if len(txt_files) > 1:
                logging.warning(f"Multiple .txt files found in steps/: {txt_files}. Using the first one.")
            steps_file = txt_files[0]
        
        logging.info(f"Found steps file: {steps_file} ✅")
        return steps_file
    except Exception as e:
        logging.error(f"Failed to find steps file: {e}")
        raise

def select_widget_task_json(domain, task_timestamp):
    """
    Generate a task.json file using LLM based on steps/**.txt and Web_Scrap/scraped_data/<domain>_widgets.txt.
    Save the task.json file in the Task folder.
    """
    try:
        widgets_txt_file = os.path.join(os.path.dirname(__file__), "Web_Scrap", "scraped_data", f"{domain}_widgets.txt")
        plan_file = find_steps_file(task_timestamp)
        task_json_path = os.path.join(os.path.dirname(__file__), "Task", "task.json")

        widgets_content = read_file(widgets_txt_file)
        steps_content = read_file(plan_file)

        system_prompt = """
You are an AI assistant tasked with creating a task.json file for selecting widgets based on availability and requirements.
Input:
- widgets_txt_file.txt: List of available widgets with IDs and attributes.
- plan_file.txt: User requirements for widget selection.

Output:
A JSON list containing a 'tasks' array, where each task has:
- action: The action to perform (e.g., "goto", "wait_for_selector", "type", "press", "click", "log").
- url, selector, text, key, timeout, message: Relevant parameters for the action.

Return ONLY the JSON list, without any explanatory text, markdown, or additional content. For example:
[
    {"action": "goto", "url": "https://www.amazon.com"},
    {"action": "wait_for_selector", "selector": "input#twotabsearchtextbox", "timeout": 1500},
    {"action": "type", "selector": "input#twotabsearchtextbox", "text": "laptop"},
    {"action": "press", "selector": "input#twotabsearchtextbox", "key": "Enter"},
    {"action": "wait_for_selector", "selector": ".s-main-slot", "timeout": 1500},
    {"action": "click", "selector": "span.a-size-base:text('64 GB')"},
    {"action": "click", "selector": "span.a-size-base:text('15 to 15.9 Inches')"},
    {"action": "wait_for_selector", "selector": "div.s-result-item", "timeout": 1500},
    {"action": "click", "selector": "div.s-result-item:first"},
    {"action": "wait_for_selector", "selector": "div#productTitle", "timeout": 1500},
    {"action": "log", "message": "Product title: {{page.title}}"},
    {"action": "click", "selector": "input#add-to-cart-button"},
    {"action": "wait_for_selector", "selector": "div#nav-cart-count", "timeout": 1000},
    {"action": "click", "selector": "div#nav-cart-count"},
    {"action": "click", "selector": "input#sc-buy-it-again-button"},
    {"action": "wait_for_selector", "selector": "div#orderSummary", "timeout": 1500},
    {"action": "log", "message": "Order summary: {{page.url}}"},
    {"action": "click", "selector": "input#placeYourOrder"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "log", "message": "Order placed successfully: {{page.url}}"}
]
"""

        user_prompt = f"""
Widgets:
{widgets_content}

Steps:
{steps_content}

Generate a task.json file as a JSON list based on the above inputs, ensuring it aligns with the steps provided and uses the available widgets.
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

                # Log and validate the raw response
                raw_response = response.choices[0].message.content
                logging.info(f"Raw LLM response: {raw_response}")
                
                if not raw_response.strip():
                    raise ValueError("Empty response from LLM")

                # Extract JSON from the response
                task_json = extract_json_from_response(raw_response)

                # Validate the JSON structure
                if not isinstance(task_json, list) or not all(isinstance(task, dict) for task in task_json):
                    raise ValueError("Invalid task.json format: Expected a list of task dictionaries")

                write_json(task_json_path, task_json)
                logging.info(f"Generated task.json at {task_json_path} ✅")
                return task_json_path
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to parse LLM response after {max_retries} attempts: {raw_response}")
                time.sleep(5)  # Increased delay to avoid rate limits
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
            ["python", "Automate_action/action_playwright.py"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"action_playwright.py failed: {result.stderr}")

        url = result.stdout.strip()
        if not url.startswith("http"):
            raise ValueError(f"Invalid URL: {url}")

        domain = extract_domain(url)
        logging.info(f"action_playwright.py output URL: {url}, domain: {domain} ✅")
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
        if len(csv_files) > 1:
            logging.warning(f"Multiple CSV files found: {csv_files}. Using the first one.")
        csv_file = csv_files[0]
        logging.info(f"Found CSV file: {csv_file} ✅")
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
            ["python", "Web_Scrap/product.py", url, domain],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"product.py failed: {result.stderr}")

        csv_file = find_csv_file()
        logging.info(f"product.py completed, found generated CSV ✅")
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
        if not all(col in df.columns for col in ['title', 'price', 'rating', 'review', 'URL']):
            raise ValueError("CSV missing required columns: [title, price, rating, review, URL]")
        products = df.to_dict('records')

        plan_file = find_steps_file()
        steps_content = read_file(plan_file)

        system_prompt = """
You are an AI assistant tasked with selecting a product from a CSV file and creating a task.json to navigate to its URL, add it to the cart, and take a screenshot.
Input:
- CSV: Contains [title, price, rating, review, URL] for top 10 products.
- plan_file.txt: Description of the desired product.

Output:
A JSON list with a 'tasks' array, where tasks include:
- action: The action to perform (e.g., "goto", "click", "screenshot").
- url: The product URL (for goto).
- selector: CSS selector for the add-to-cart button (if applicable).
- path: File path for screenshot (if applicable).
- timeout: Timeout for waiting actions (if applicable).

Return ONLY the JSON list, without any explanatory text or markdown. For example:
[
    {"action": "goto", "url": "https://example.com/product"},
    {"action": "wait_for_selector", "selector": "button.add-to-cart", "timeout": 1500},
    {"action": "click", "selector": "button.add-to-cart"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "screenshot", "path": "cart_added.png"},
    {"action": "log", "message": "Product added to cart: {{page.url}}"}
]
"""

        user_prompt = f"""
Products:
{json.dumps(products, indent=2)}

Steps:
{steps_content}

Select the product that best matches the description in plan_file.txt and generate a task.json to navigate to its URL, add it to the cart, and take a screenshot.
"""

        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
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
        logging.info(f"Generated task.json for product selection and cart addition at {task_json_path} ✅")
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
            ["python", "Automate_action/action_playwright.py"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"action_playwright.py for cart failed: {result.stderr}")

        output = result.stdout.strip()
        logging.info(f"action_playwright.py for cart completed: {output} ✅")
        print(f"Chosen product added to cart: {output}")
    except Exception as e:
        logging.error(f"Failed to run action_playwright.py for cart: {e}")
        raise

def main(task_timestamp=None):
    """Main function to orchestrate the workflow."""
    try:
        # Step 1: Extract domain from the steps file or assume from task context
        plan_file = find_steps_file(task_timestamp)
        steps_content = read_file(plan_file)
        # Assuming the URL is mentioned in steps_content or provided earlier
        url_match = re.search(r'https?://[^\s]+', steps_content)
        if not url_match:
            raise ValueError("No URL found in steps file")
        url = url_match.group(0)
        domain = extract_domain(url)
        logging.info(f"Starting workflow with domain: {domain} ✅")

        # Step 2: Generate first task.json using LLM
        select_widget_task_json(domain, task_timestamp)

        # Step 3: Run action_playwright.py to get filtered URL
        url, domain = run_action_playwright()

        # Step 4: Run product.py to fetch top 10 products
        time.sleep(1)  # Ensure previous step completes
        csv_file = run_product_scraper(url, domain)

        # Step 5: Select product and generate second task.json
        select_product_and_generate_task_json(csv_file)

        # Step 6: Run action_playwright.py to add product to cart
        run_action_playwright_for_cart()

        logging.info("Workflow completed successfully ✅")
        return {"status": "success", "message": "Workflow completed"}
    except Exception as e:
        logging.error(f"Workflow failed: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import sys
    task_timestamp = sys.argv[1] if len(sys.argv) > 1 else None
    result = main(task_timestamp)
    print(json.dumps(result))  # Output result for main.py to capture
'''
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
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, response_text)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)
        return json.loads(response_text)
    except json.JSONDecodeError as e:
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

        system_prompt = """
You are an AI assistant tasked with creating a task.json file for 
selecting widgets based on availability and requirements.
Input:
- widgets_txt_file.txt: List of available widgets with IDs and attributes.
- plan_file.txt: User requirements for widget selection.

Output:
A JSON object with a 'tasks' array, where each task has:
- action: The action to perform (e.g., "select_widget", "confirm_selection").
- widget_id: The ID of the widget to select.
- criteria: The selection criteria (e.g., "available").

Example:
        [
    {"action": "goto", "url": "https://www.amazon.com"},
    {"action": "log", "message": "After goto: URL is {{page.url}}"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "wait_for_selector", "selector": "input#twotabsearchtextbox", "timeout": 1500},
    {"action": "focus", "selector": "input#twotabsearchtextbox"},
    {"action": "type", "selector": "input#twotabsearchtextbox", "text": "HP laptop"},
    {"action": "press", "selector": "input#twotabsearchtextbox", "key": "Enter"},
    {"action": "blur", "selector": "input#twotabsearchtextbox"},
    {"action": "wait_for_selector", "selector": ".s-main-slot", "timeout": 1500},
    {"action": "wait_for_selector", "selector": "#s-refinements", "timeout": 1500},
    {"action": "evaluate", "selector": "#s-refinements", "script": "el => el.scrollTop = 0"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "scroll_into_view", "selector": "span.a-size-base:text('HP')"},
    {"action": "hover", "selector": "span.a-size-base:text('HP')"},
    {"action": "click", "selector": "span.a-size-base:text('HP')"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "log", "message": "Applied filter: HP"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "evaluate", "selector": "#s-refinements", "script": "el => el.scrollTop = 0"},
    {"action": "scroll_into_view", "selector": "span.a-size-base:text('64 GB')"},
    {"action": "hover", "selector": "span.a-size-base:text('64 GB')"},
    {"action": "click", "selector": "span.a-size-base:text('64 GB')"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "log", "message": "Applied filter: 64 GB RAM"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "evaluate", "selector": "#s-refinements", "script": "el => el.scrollTop = 0"},
    {"action": "scroll_into_view", "selector": "span.a-size-base:text('17 Inches & Above'), span.a-size-base:text('16 to 16.9 Inches')"},
    {"action": "hover", "selector": "span.a-size-base:text('17 Inches & Above'), span.a-size-base:text('16 to 16.9 Inches')"},
    {"action": "click", "selector": "span.a-size-base:text('17 Inches & Above'), span.a-size-base:text('16 to 16.9 Inches')"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "log", "message": "Applied filter: >16 Inch Display"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "evaluate", "selector": "#s-refinements", "script": "el => el.scrollTop = 0"},
    {"action": "scroll_into_view", "selector": "span.a-size-base:text('4 TB & Above'), span.a-size-base:text('3 TB'), span.a-size-base:text('2 TB'), span.a-size-base:text('1.5 TB')"},
    {"action": "hover", "selector": "span.a-size-base:text('4 TB & Above'), span.a-size-base:text('3 TB'), span.a-size-base:text('2 TB'), span.a-size-base:text('1.5 TB')"},
    {"action": "click", "selector": "span.a-size-base:text('4 TB & Above'), span.a-size-base:text('3 TB'), span.a-size-base:text('2 TB'), span.a-size-base:text('1.5 TB')"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "log", "message": "Applied filter: Hard Drive >1TB"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "evaluate", "selector": "#s-refinements", "script": "el => el.scrollTop = 0"},
    {"action": "scroll_into_view", "selector": "span.a-size-base:text-matches('Linux|Ubuntu', 'i')"},
    {"action": "hover", "selector": "span.a-size-base:text-matches('Linux|Ubuntu', 'i')"},
    {"action": "click", "selector": "span.a-size-base:text-matches('Linux|Ubuntu', 'i')"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "log", "message": "Applied filter: Linux/Ubuntu"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "evaluate", "selector": "#s-refinements", "script": "el => el.scrollTop = 0"},
    {"action": "scroll_into_view", "selector": "span.a-size-base:text('New')"},
    {"action": "hover", "selector": "span.a-size-base:text('New')"},
    {"action": "click", "selector": "span.a-size-base:text('New')"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "log", "message": "Applied filter: New"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "evaluate", "selector": "#s-refinements", "script": "el => el.scrollTop = 0"},
    {"action": "scroll_into_view", "selector": "span.a-size-base:text('4.0 GHz & Above')"},
    {"action": "hover", "selector": "span.a-size-base:text('4.0 GHz & Above')"},
    {"action": "click", "selector": "span.a-size-base:text('4.0 GHz & Above')"},
    {"action": "wait_for_timeout", "timeout": 2000},
    {"action": "log", "message": "Applied filter: 4.0 GHz & Above"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "screenshot", "path": "filters_applied.png"},
    {"action": "log", "message": "Captured filters screenshot"},
    {"action": "wait_for_timeout", "timeout": 1000},
    {"action": "wait_for_selector", "selector": "div[data-component-type='s-search-result']", "timeout": 1500},
    {"action": "evaluate", "script": "document.querySelectorAll('div[data-component-type=\"s-search-result\"]').length > 0 ? 'Results found' : 'No results, filters applied'"},
    {"action": "log", "message": "Search results status: {{evaluate_result}}"},
    {"action": "log", "message": "Final search results URL: {{page.url}}"}
]

    Remember the above provided example is just for reference for you, 
    you are not bound to strictly copy the example okay. only use it 
    as reference..
    
    Remember only selecting the widgets is your work do nothing other than that, whatever widgets
    are been told to choose select/click them and then give me the url in the end.
    and keep the delay between each step to be 1sec
    
    
    """

        user_prompt = f"""
Widgets:
{widgets_content}

Steps:
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

        url = result.stdout.strip()
        if not url.startswith("http"):
            raise ValueError(f"Invalid URL: {url}")

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
        if not all(col in df.columns for col in ['title', 'price', 'rating', 'review', 'URL']):
            raise ValueError("CSV missing required columns: [title, price, rating, review, URL]")
        products = df.to_dict('records')

        plan_file = find_steps_file()
        steps_content = read_file(plan_file)

        system_prompt = """
You are an AI assistant tasked with selecting a product from a CSV file and creating a task.json to navigate to its URL, add it to the cart, and take a screenshot.
Input:
- CSV: Contains [title, price, rating, review, URL] for top 10 products.
- plan_file.txt: Description of the desired product.

Output:
A JSON list with a 'tasks' array, where tasks include:
- action: The action to perform (e.g., "goto", "click", "screenshot").
- url: The product URL (for goto).
- selector: CSS selector for the add-to-cart button (if applicable).
- path: File path for screenshot (if applicable).
- timeout: Timeout for waiting actions (if applicable).

Return ONLY the JSON list, without any explanatory text or markdown.
"""

        user_prompt = f"""
Products:
{json.dumps(products, indent=2)}

Steps:
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