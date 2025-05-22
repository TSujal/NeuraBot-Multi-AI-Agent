# 🧠 NeuraBot: An Intelligent Multi-Agent LLM Chatbot with Real-Time RAG and Task Execution

NeuraBot is a next-generation intelligent chatbot framework designed for advanced task automation, dynamic planning, and real-time web interaction — powered by multi-agent systems, RAG pipelines, and Playwright-based web automation.

---

## 🚀 Key Features

### 🤖 Chatbot with Integrated RAG
- Built on **LLaMA-3.3-8B**, the primary chatbot uses **Retrieval-Augmented Generation (RAG)** to enrich conversations with up-to-date information from:
  - Wikipedia
  - Google Search
  - arXiv, PubMed, Reddit, GitHub, and more
- Supports both **text** and **speech** input, enhancing user accessibility.

### ✅ Dual Chat Modes via Streamlit UI
- A checkbox in the **Streamlit interface** switches between:
  - **Standard Chatbot Mode** (RAG-powered)
  - **Task-Driven Agent Mode** (automation environment)

---

## 🛠️ Task Automation Workflow

In Task Mode, NeuraBot transforms into an intelligent assistant capable of performing real-time actions on websites like **Amazon** or **Walmart**:

### 🧩 Step-by-Step Multi-Agent Pipeline

1. **Instruction Agent**
   - Takes initial user task (e.g., “Find a budget laptop under $600 with SSD”).
   - Asks follow-up questions until it has enough clarity.
   - Generates a high-level task plan in JSON format.

2. **Widget Scraper Agent**
   - Navigates to the specified site (e.g., Amazon).
   - Scrapes page UI elements (filters, buttons, categories) using Playwright.

3. **Planning Agent**
   - Combines `{scraped_widgets}` + `{user_tasks}`.
   - Generates detailed Playwright action plans for webpage interaction.

4. **Product Scraper Agent**
   - Executes the plan to collect top 10 matching products.
   - Saves product details to a `.csv` file.

5. **Manager Agent**
   - Matches `{user_preferences}` with `{product_data.csv}`.
   - Selects the best match and creates a new action plan.

6. **Automation Agent**
   - Uses Playwright + Groq + LLaMA-3.3-70B to:
     - Add product to cart
     - Take screenshots
     - Seamlessly complete interactions

---

## 🧠 Technologies Used

| Component | Tech Stack |
|----------|-------------|
| Language Models | LLaMA 3.3 (8B & 70B), DeepSeek R1 |
| Frameworks | Streamlit, Playwright |
| Agents & RAG | LangGraph, DeepSeek-R1, Crawl4AI |
| Tooling | Groq API, Python, JSON Planning, CSV |
| Search Integrations | Google, Wikipedia, arXiv, PubMed, GitHub |
| Automation | Playwright-driven browser automation |

---

## 🧪 Project Showcase

### ✅ Streamlit UI
- Real-time UI mode switching (chat ↔️ task).
- Interactive follow-up for task clarification.
- Visual feedback loops and screen captures for transparency.

---

## 📦 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/TSujal/NeuraBot.git
cd NeuraBot

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run main.py
