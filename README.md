# 🔍 Multi-Agent Smart Chatbot with Task Planning & Web Automation

This project is a multi-functional AI system that combines a **standard chatbot with RAG (Retrieval-Augmented Generation)** and an **autonomous task execution assistant** capable of interacting with live websites (like Amazon) to perform complex tasks based on user input and contextual understanding.

---

## 🌟 Features

### 🧠 Chatbot with RAG
- Integrates Retrieval-Augmented Generation using sources like:
  - Wikipedia
  - Google Search
  - Additional data endpoints (customizable)
- Offers intelligent conversational abilities with contextual information gathering.

### ✅ Checkbox-Enabled Task Agent
When the checkbox in the Streamlit UI is activated, the system transitions into a **task planning mode**:
- The chatbot turns into a **task manager agent**.
- It doesn't immediately act — instead, it:
  - Asks a series of **follow-up clarification questions**.
  - Gathers specific details about what the user wants to accomplish.
  - Builds a complete understanding before initiating any actions.

---

## 🧩 Agent Workflow Overview

### 1. 🗺️ Task Clarification Phase
- The **planning agent** keeps asking clarifying questions based on user input.
- Once satisfied with the answers, it proceeds to generate a **task plan** in structured JSON format.

### 2. 🌐 Widget Scraping
- A **widget scraping agent** uses tools (like Playwright) to:
  - Access the provided webpage (e.g., Amazon).
  - Scrape all **widgets, filters, and UI elements** visible for the target product.

### 3. 📑 Action Plan Generation
- The scraped widget data + task data is given to a **planning agent**.
- This agent:
  - Creates an **action sequence** for selecting widgets and applying filters on the screen using the Playwright automation library.

### 4. 📦 Product Scraping Agent
- A dedicated **product scraper agent**:
  - Uses the selected filters and site context to fetch the **top 10 matching products**.
  - Stores this product list in a `CSV` file with relevant metadata.

### 5. 🤖 Match & Decision Agent
- The **manage agent** compares:
  - `{user_given_info}` with `{product_Scrap.csv}`
- It selects the most appropriate product based on match criteria.

### 6. 🛒 Final Automation & Checkout
- The selected product is:
  - Re-fed into the **action agent**.
  - The agent navigates and **adds the product to the cart**.
  - A **screenshot** of the final state is taken as confirmation.

---

## ⚙️ Tech Stack

- **Python**
- **Streamlit** for UI
- **LangChain Agents**
- **Playwright** for browser automation
- **RAG pipelines** (Wikipedia, Google Search)
- **Multi-Agent Orchestration**
- **CSV for result tracking**

---

## 🚀 Future Improvements

- Add support for multiple e-commerce platforms (Flipkart, Myntra, etc.)
- Automate user authentication & checkout steps
- Enhance LLM reasoning with feedback loop integration
- Store historical plans & decisions for training fine-tuned models

---

## 💡 Inspiration

This project was built to demonstrate how autonomous AI agents can **think, ask questions, and act** just like humans — combining **intelligence, planning, and execution** into one seamless system.
---

## 🧪 How to Run

```bash
# Clone repo
git clone https://github.com/TSujal/NeuraBot-Multi-AI-Agent

# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run main.py

