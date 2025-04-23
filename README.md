# Neurabot Multi AI Agent
A multi-agent AI system for web scraping, automation, and task management using Playwright and Python.

## Installation
1. Clone: `git clone https://github.com/TSujal/NeuraBot-Multi-AI-Agent.git`
2. Create venv: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install: `pip install -r requirements.txt`

## Usage
Run: `python main.py`

## Project Structure
- `Automate_action/`: Playwright automation (`action_playwright.py`, `filters_applied.png`)
- `Task/`: Task configurations (`task.json`)
- `Web_Scrap/`: Scraping logic (`products.py`, `widgets.py`)
- `site_config_product/`: Product configs (`amazon.json`, `walmart.json`)
- `site_config_widget/`: Widget configs (`amazon.json`, `walmart.json`)
- `steps/`: Workflow steps (`task_plan_20250421_191753.txt`)
- `main.py`, `manage.py`: Core scripts
- `test_*.py`: Test scripts

## Contributing
Open issues or pull requests.