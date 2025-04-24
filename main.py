import streamlit as st # type: ignore
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper # type: ignore
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun # type: ignore
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler # type: ignore
from langchain.agents import initialize_agent, AgentType
import os
from dotenv import load_dotenv
import json
import logging
import re
import datetime
import os.path
import time
import glob
import subprocess

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.title("NeuraBot Search Engine with Agent Integration")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API KEY:", type="password", value=os.getenv("GROQ_API_KEY", ""))
enable_ai_agent = st.sidebar.checkbox("Enable AI Agent", value=False)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am NeuraBot, a chatbot who can search the web or help with tasks. How can I assist you today?"}
    ]
if "reflection_data" not in st.session_state:
    st.session_state["reflection_data"] = []
if "feedback" not in st.session_state:
    st.session_state["feedback"] = []
if "task_context" not in st.session_state:
    st.session_state["task_context"] = {
        "task": None,
        "item": None,
        "details": {},
        "questions_asked": [],
        "question_types_asked": [],
        "question_count": 0,
        "confirmation_asked": False
    }
if "prev_ai_agent_state" not in st.session_state:
    st.session_state["prev_ai_agent_state"] = enable_ai_agent

# Clear screen if checkbox state changes
if st.session_state["prev_ai_agent_state"] != enable_ai_agent:
    st.session_state["messages"] = [
        {"role": "assistant", "content": f"{'AI Agent' if enable_ai_agent else 'Default'} mode activated. I’m happy to assist—how can I help you today?"}
    ]
    st.session_state["reflection_data"] = []
    st.session_state["feedback"] = []
    st.session_state["task_context"] = {
        "task": None,
        "item": None,
        "details": {},
        "questions_asked": [],
        "question_types_asked": [],
        "question_count": 0,
        "confirmation_asked": False
    }
    st.session_state["prev_ai_agent_state"] = enable_ai_agent
    st.rerun()

# Display chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and not enable_ai_agent:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", key=f"thumbs_up_{idx}"):
                    st.session_state.feedback.append({"message_idx": idx, "rating": "positive"})
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎", key=f"thumbs_down_{idx}"):
                    st.session_state.feedback.append({"message_idx": idx, "rating": "negative"})
                    st.warning("Thanks for your feedback! We'll try to improve.")

def reflect_on_response(llm_model, prompt: str, initial_response: str, tools_used: list) -> dict:
    reflection_prompt = (
        "You are a reflective AI assistant tasked with evaluating and improving a response to a user's query. "
        "Your goal is to ensure the response is accurate, relevant, complete, and clear. "
        f"**User Prompt**: {prompt}\n"
        f"**Initial Response**: {initial_response}\n"
        f"**Tools Used**: {', '.join([tool.name for tool in tools_used])}\n\n"
        "Evaluate based on accuracy, relevance, completeness, and clarity. "
        "Identify strengths and weaknesses. "
        "If needed, provide a refined response. "
        "Return JSON:\n"
        "```json\n"
        "{\n"
        "  \"evaluation\": {\"accuracy\": \"\", \"relevance\": \"\", \"completeness\": \"\", \"clarity\": \"\"},\n"
        "  \"strengths\": \"\",\n"
        "  \"weaknesses\": \"\",\n"
        "  \"needs_refinement\": false,\n"
        "  \"refined_response\": \"\"\n"
        "}\n"
        "```"
    )
    try:
        response = llm_model.invoke(reflection_prompt)
        reflection_result = json.loads(response.content)
        logging.info(f"Reflection result: {reflection_result}")
        return reflection_result
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse reflection response: {e}")
        return {
            "evaluation": {"accuracy": "Unknown", "relevance": "Unknown", "completeness": "Unknown", "clarity": "Unknown"},
            "strengths": "Could not evaluate",
            "weaknesses": "Reflection parsing failed",
            "needs_refinement": False,
            "refined_response": initial_response
        }

def process_default_input(prompt: str, llm_model, tools: list) -> str:
    search_agent = initialize_agent(
        tools,
        llm_model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )
    with st.spinner("Generating response..."):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        initial_response = search_agent.run(
            f"Use available tools to provide a detailed answer to: {prompt}",
            callbacks=[st_cb]
        )
        logging.info(f"Initial response: {initial_response}")
    with st.spinner("Refining response..."):
        reflection_result = reflect_on_response(llm_model, prompt, initial_response, tools)
        st.session_state.reflection_data.append({
            "prompt": prompt,
            "initial_response": initial_response,
            "reflection": reflection_result,
            "model_used": "Mistral-Saba-24b"
        })
    final_response = reflection_result["refined_response"] if reflection_result["needs_refinement"] and reflection_result["refined_response"] else initial_response
    return final_response

def clean_product_name(item: str) -> str:
    """Clean the product name by removing 'for me', articles, and excessive whitespace."""
    if not item:
        return ""
    # Remove 'for me', articles ('a', 'an', 'the'), and normalize whitespace
    cleaned = re.sub(r'\b(for\s+me|a|an|the)\b', '', item, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

def reset_task_context():
    return {
        "task": None,
        "item": None,
        "details": {},
        "questions_asked": [],
        "question_types_asked": [],
        "question_count": 0,
        "confirmation_asked": False
    }

def save_to_file(steps: list, task_description: str) -> str:
    os.makedirs("steps", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"steps/task_plan_{timestamp}.txt"
    output = "Task Plan:\n"
    for i, step in enumerate(steps, 1):
        output += f"{i}. {step}\n"
    output += f"\nTask Description: {task_description}\n"
    with open(filename, "w") as f:
        f.write(output)
    return filename

def get_next_question_type(task_context: dict) -> tuple[str, str]:
    """Select the next unasked question type and generate a task-specific question."""
    cleaned_item = clean_product_name(task_context["item"])
    available_types = [
        ("source", f"Could you tell me where you’d like to order your {cleaned_item} from? (e.g., a specific store or website)"),
        ("specs", f"What specific features are you looking for in your {cleaned_item}?"),
        ("budget", f"What’s your budget for the {cleaned_item}?"),
        ("delivery", f"Would you like delivery or pickup for the {cleaned_item}?"),
        ("timing", f"When would you like the {cleaned_item} to be ordered or delivered?")
    ]
    for q_type, question in available_types:
        if q_type not in task_context["question_types_asked"]:
            return q_type, question
    return "none", "Is there anything else you’d like to add before I create the plan?"

def extract_timestamp_from_filename(filename: str) -> str:
    """Extract the timestamp from a steps file name (e.g., task_plan_20250421_165450.txt -> 20250421_165450)."""
    match = re.search(r'task_plan_(\d{8}_\d{6})\.txt', os.path.basename(filename))
    if match:
        return match.group(1)
    return ""

def process_task_plan_file(filename: str, api_key: str, task_item: str) -> str:
    """Process the task plan file with LLM and return the formatted output."""
    try:
        with open(filename, "r") as f:
            task_plan_content = f.read()
        logging.info(f"Successfully read content from {filename}")
        
        llm_model = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-70b-8192",
            streaming=True,
            max_tokens=100
        )
        
        cleaned_item = clean_product_name(task_item)
        llm_prompt = (
            "You are an AI assistant tasked with analyzing a task plan. "
            "Below is the content of a task plan file:\n\n"
            f"{task_plan_content}\n\n"
            f"Task Item from Context: {cleaned_item}\n\n"
            "From the first two steps, identify the website/URL (e.g., amazon, walmart) and the product being searched for. "
            "Return the answer in the EXACT format: <domain_name> <product_name>\n"
            "Use the item name from the task context as the product_name, keeping it simple and without specifications. "
            "For example, if the task item is 'sports shoes' and step 1 is 'Navigate to Amazon's website at https://www.amazon.com' "
            "and step 2 is 'Search for sports shoes men size 9', the output MUST be: amazon sports shoes\n"
            "Do NOT include any additional text, explanations, or punctuation like commas in the product_name. "
            "The product_name should be a simple phrase without specifications like size or specific model names."
        )
        
        with st.spinner("Analyzing task plan with LLM..."):
            llm_response = llm_model.invoke(llm_prompt)
            output = llm_response.content.strip()
            logging.info(f"LLM output: {output}")
        
        # Relaxed regex to allow apostrophes and hyphens
        if not re.match(r'^\w+\s+[\w\s\'-]+$', output):
            logging.warning(f"Invalid LLM output format: {output}. Falling back to task context item.")
            # Extract domain (first word) and use cleaned task item
            domain = output.split()[0] if output and output.split() else "unknown"
            if cleaned_item:
                output = f"{domain} {cleaned_item}"
            else:
                raise ValueError(f"Invalid LLM output and no valid task item: {output}")
        
        return output
    
    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
        # Fallback to task context item if available
        cleaned_item = clean_product_name(task_item)
        if cleaned_item:
            domain = "amazon"  # Default to amazon if domain extraction fails
            return f"{domain} {cleaned_item}"
        return f"Failed to extract task info: {e}"

def run_manage_py(task_timestamp: str) -> dict:
    """Run manage.py with the given task timestamp and return its JSON output."""
    try:
        result = subprocess.run(
            ["python", "manage.py", task_timestamp],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"manage.py executed with return code: {result.returncode}")
        
        # Parse the JSON output from manage.py
        try:
            output = json.loads(result.stdout)
            logging.info(f"manage.py output: {output}")
            return output
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse manage.py output: {e}, Raw output: {result.stdout}")
            return {"status": "error", "message": f"Invalid JSON output from manage.py: {e}"}
    except subprocess.CalledProcessError as e:
        logging.error(f"manage.py failed: {e.stderr}")
        return {"status": "error", "message": f"manage.py failed with error: {e.stderr}"}
    except Exception as e:
        logging.error(f"Unexpected error running manage.py: {e}")
        return {"status": "error", "message": f"Unexpected error: {e}"}

def process_ai_agent_input(prompt: str, llm_model, system_prompt: str, api_key: str) -> dict:
    task_context = st.session_state.task_context
    task_keywords = ['order', 'buy', 'purchase', 'plan', 'book', 'get', 'make', 'pizza', 'food', 'meal', 'grocery', 'laptop', 'phone', 'party', 'trip', 'shoe', 'sneakers']
    order_pattern = re.match(r'^(order|buy)\s+(?:(?:for\s+me\s+)(?:a|an|the\s+)?)?([\w\s]+?)(?:\s+from\s+([\w\s]+))?$', prompt.lower().strip())
    conversational_keywords = ['thanks', 'thank you', 'cool', 'great', 'awesome', 'looks good']
    
    is_conversational = any(keyword in prompt.lower() for keyword in conversational_keywords)
    is_task = order_pattern or any(keyword in prompt.lower() for keyword in task_keywords)
    is_confirmation_response = task_context["confirmation_asked"] and prompt.lower() in ["no", "nothing", "no nothing", "nothing else", "looks good"]
    
    if is_conversational and not is_task and not is_confirmation_response:
        if task_context["task"]:
            return {
                "status": "conversational",
                "question": "You’re very welcome! Anything else I can help with?",
                "steps": [],
                "task_description": ""
            }
        else:
            st.session_state.task_context = reset_task_context()
            return {
                "status": "conversational",
                "question": "You’re very welcome! How can I assist you today?",
                "steps": [],
                "task_description": ""
            }
    
    if is_task and (not task_context["task"] or task_context["details"].get("steps_provided")):
        st.session_state.task_context = reset_task_context()
        task_context = st.session_state.task_context
        task_context["task"] = prompt
        if order_pattern:
            task_context["item"] = order_pattern.group(2).strip()
            if order_pattern.group(3):
                task_context["details"]["source"] = order_pattern.group(3).strip()
    
    history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages[-10:]])
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Task Context: {json.dumps(task_context)}\n"
        f"Conversation History:\n{history}\n\n"
        f"User Input: {prompt}\n"
        f"Previously Asked Questions: {', '.join(task_context['questions_asked']) or 'None'}"
    )
    
    logging.info(f"Input classification: {'Task' if is_task else 'Conversational'}, Confirmation: {is_confirmation_response}, Input: {prompt}, Task Context: {json.dumps(task_context)}")
    
    common_providers = ['amazon', 'walmart', 'best buy', 'nike', 'adidas', 'target', 'uber eats', 'doordash']
    is_source_response = (
        prompt.lower() in common_providers or
        (len(prompt.split()) <= 3 and any("source" in q_type for q_type in task_context["question_types_asked"]))
    )
    
    # Path to the widgets.py script
    script_path = os.path.join("Web_Scrap", "widgets.py")
    
    try:
        response = llm_model.invoke(full_prompt)
        raw_response = response.content.strip()
        logging.info(f"Raw AI Agent response: {raw_response}")
        
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if not json_match:
            logging.error("No valid JSON found in response")
            raise json.JSONDecodeError("No JSON object found", raw_response, 0)
        json_str = json_match.group(0)
        logging.info(f"Extracted JSON: {json_str}")
        
        result = json.loads(json_str)
        logging.info(f"Parsed AI Agent response: {result}")
        
        if not all(key in result for key in ["status", "question", "steps", "task_description"]):
            logging.error("Invalid JSON structure: missing required keys")
            if not task_context["task"]:
                task_context["task"] = prompt
                task_context["item"] = order_pattern.group(2).strip() if order_pattern else None
                question_type, question = get_next_question_type(task_context)
                task_context["questions_asked"].append(question)
                if question_type != "none":
                    task_context["question_types_asked"].append(question_type)
                task_context["question_count"] += 1
                st.session_state.task_context = task_context
                return {
                    "status": "clarification_needed",
                    "question": question,
                    "steps": [],
                    "task_description": ""
                }
            if is_source_response:
                task_context["details"]["source"] = prompt.lower()
                question_type, question = get_next_question_type(task_context)
                task_context["questions_asked"].append(question)
                if question_type != "none":
                    task_context["question_types_asked"].append(question_type)
                task_context["question_count"] += 1
                st.session_state.task_context = task_context
                return {
                    "status": "clarification_needed",
                    "question": question,
                    "steps": [],
                    "task_description": ""
                }
            if task_context["confirmation_asked"]:
                if is_confirmation_response:
                    specs = task_context["details"].get("specs", "")
                    source = task_context["details"].get("source", "the specified service")
                    source_url = f"https://www.{source}.com" if source in common_providers else f"https://www.{source}.com"
                    product_name = clean_product_name(task_context["item"])
                    steps = [
                        f"Open a web browser like Google Chrome or Firefox.",
                        f"Navigate to {source}'s website at {source_url}.",
                        f"Prepare to search for the product: {product_name}.",
                        f"Locate the search bar, usually at the top of the {source} homepage.",
                        f"Click the search bar and type '{product_name}'.",
                        f"Press Enter or click the search icon to start the search.",
                        f"Wait for the search results to load completely.",
                        f"If available, apply filters based on your specifications: {specs}.",
                        f"Wait for the page to reload after applying filters.",
                        f"Sort the results by relevance or customer reviews if available.",
                        f"Look through the search results to find a {product_name} that matches your specifications: {specs}.",
                        f"Select a {product_name} by clicking its title or image to visit its product page.",
                        f"Verify the product details on the page to confirm they match your specifications: {specs}.",
                        f"Check the price to ensure it fits within your budget: {task_context['details'].get('budget', 'no budget specified')}.",
                        f"Locate the 'Add to Cart' button on the product page.",
                        f"Click 'Add to Cart' to add the {product_name} to your cart.",
                        f"Wait for a confirmation that the {product_name} was added to your cart.",
                        f"If a warranty or additional offer pop-up appears, select 'No Thanks' or decline.",
                        f"If a CAPTCHA appears, follow the on-screen instructions to complete it.",
                        f"Click the cart icon, usually in the top-right corner, to view your cart.",
                        f"Verify that the {product_name} in the cart matches your specifications: {specs}.",
                        f"If the item is incorrect, click 'Remove' to delete it from the cart.",
                        f"Click 'Proceed to Checkout' to start the checkout process.",
                        f"If prompted to sign in, enter your email and password to log in.",
                        f"Select the delivery option '{task_context['details'].get('delivery', 'standard')}'.",
                        f"Enter or confirm your delivery address.",
                        f"Review the order summary to ensure the {product_name} and total cost are correct.",
                        f"Click 'Place Order' or 'Confirm Order' to complete the purchase.",
                        f"Wait for the order confirmation page and note the order number or take a screenshot.",
                        f"Close the browser."
                    ]
                    budget_info = task_context['details'].get('budget', 'no budget specified')
                    task_description = (
                        f"The task is to order a {product_name} from {source} "
                        f"with the following specifications: {specs or 'no specific preferences provided'}. "
                        f"The order will be completed with {task_context['details'].get('delivery', 'standard')} delivery, "
                        f"and the budget is {budget_info}. "
                        f"The order is to be placed as soon as possible unless otherwise specified."
                    )
                    filename = save_to_file(steps, task_description)
                    task_context["details"]["steps_provided"] = True
                    st.session_state.task_context = reset_task_context()
                    
                    # Process the saved task plan file with LLM
                    time.sleep(1)
                    steps_folder = "steps"
                    txt_files = glob.glob(os.path.join(steps_folder, "*.txt"))
                    if txt_files:
                        txt_files.sort(key=os.path.getctime, reverse=True)
                        first_txt_file = txt_files[0]
                        logging.info(f"Selected first .txt file: {first_txt_file}")
                        
                        task_timestamp = extract_timestamp_from_filename(first_txt_file)
                        logging.info(f"Extracted task timestamp: {task_timestamp}")
                        
                        output = process_task_plan_file(first_txt_file, api_key, task_context.get("item", ""))
                        logging.info(f"Task info extracted: {output}")
                        st.session_state.messages.append({"role": "assistant", "content": output})
                        
                        # Run widgets.py with the LLM output as input argument
                        input_argument = output
                        try:
                            subprocess.run(["python", script_path, input_argument], check=True)
                            logging.info(f"Successfully ran widgets.py with input: {input_argument}")
                            
                            # Run manage.py with the task timestamp
                            with st.spinner("Thinking <running manage.py file...>"):
                                manage_result = run_manage_py(task_timestamp)
                                if manage_result["status"] == "error":
                                    logging.error(f"manage.py failed: {manage_result['message']}")
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": f"Error running task: {manage_result['message']}"
                                    })
                                else:
                                    logging.info("manage.py completed successfully ✅")
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": "Task completed successfully!"
                                    })
                        
                        except subprocess.CalledProcessError as e:
                            logging.error(f"Error running widgets.py: {e}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Error running widgets.py: {e}"
                            })
                        
                        return {
                            "status": "plan_provided",
                            "question": "",
                            "steps": steps,
                            "task_description": task_description
                        }
                    else:
                        logging.warning("No .txt files found in steps/ folder")
                        output = "No task plan files found in steps/ folder"
                        logging.info(f"Task info extracted: {output}")
                        st.session_state.messages.append({"role": "assistant", "content": output})
                        
                        # Run widgets.py with the error output as input argument
                        input_argument = output
                        try:
                            subprocess.run(["python", script_path, input_argument], check=True)
                            logging.info(f"Successfully ran widgets.py with input: {input_argument}")
                            
                            # Run manage.py with no timestamp (fallback)
                            with st.spinner("Thinking <running manage.py file...>"):
                                manage_result = run_manage_py("")
                                if manage_result["status"] == "error":
                                    logging.error(f"manage.py failed: {manage_result['message']}")
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": f"Error running task: {manage_result['message']}"
                                    })
                                else:
                                    logging.info("manage.py completed successfully ✅")
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": "Task completed successfully!"
                                    })
                        
                        except subprocess.CalledProcessError as e:
                            logging.error(f"Error running widgets.py: {e}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Error running widgets.py: {e}"
                            })
                        
                        return {
                            "status": "plan_provided",
                            "question": "",
                            "steps": steps,
                            "task_description": task_description
                        }
                else:
                    task_context["details"]["specs"] = f"{task_context['details'].get('specs', '')}, {prompt.lower()}".strip(", ")
                    task_context["confirmation_asked"] = True
                    task_context["confirmation_count"] = task_context.get("confirmation_count", 0) + 1
                    st.session_state.task_context = task_context
                    return {
                        "status": "confirmation_needed",
                        "question": "Is there anything else you’d like to add before I create the plan?",
                        "steps": [],
                        "task_description": ""
                    }
            question_type, question = get_next_question_type(task_context)
            task_context["questions_asked"].append(question)
            if question_type != "none":
                task_context["question_types_asked"].append(question_type)
            task_context["question_count"] += 1
            st.session_state.task_context = task_context
            return {
                "status": "clarification_needed",
                "question": question,
                "steps": [],
                "task_description": ""
            }
        
        if result["status"] == "clarification_needed" and result["question"]:
            if not task_context["task"]:
                task_context["task"] = prompt
                task_context["item"] = order_pattern.group(2).strip() if order_pattern else None
            question_type, question = get_next_question_type(task_context)
            if question not in task_context["questions_asked"]:
                task_context["questions_asked"].append(question)
                if question_type != "none":
                    task_context["question_types_asked"].append(question_type)
                task_context["question_count"] += 1
                result["question"] = question
        elif result["status"] == "confirmation_needed":
            task_context["confirmation_asked"] = True
            task_context["confirmation_count"] = task_context.get("confirmation_count", 0) + 1
        elif result["status"] == "plan_provided":
            task_context["details"]["steps_provided"] = True
            filename = save_to_file(result["steps"], result["task_description"])
            st.session_state.task_context = reset_task_context()
            
            time.sleep(1)
            steps_folder = "steps"
            txt_files = glob.glob(os.path.join(steps_folder, "*.txt"))
            if txt_files:
                txt_files.sort(key=os.path.getctime, reverse=True)
                first_txt_file = txt_files[0]
                logging.info(f"Selected first .txt file: {first_txt_file}")
                
                task_timestamp = extract_timestamp_from_filename(first_txt_file)
                logging.info(f"Extracted task timestamp: {task_timestamp}")
                
                output = process_task_plan_file(first_txt_file, api_key, task_context.get("item", ""))
                logging.info(f"Task info extracted: {output}")
                st.session_state.messages.append({"role": "assistant", "content": output})
                
                # Run widgets.py with the LLM output as input argument
                input_argument = output
                try:
                    subprocess.run(["python", script_path, input_argument], check=True)
                    logging.info(f"Successfully ran widgets.py with input: {input_argument}")
                    
                    # Run manage.py with the task timestamp
                    with st.spinner("Thinking <running manage.py file...>"):
                        manage_result = run_manage_py(task_timestamp)
                        if manage_result["status"] == "error":
                            logging.error(f"manage.py failed: {manage_result['message']}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Error running task: {manage_result['message']}"
                            })
                        else:
                            logging.info("manage.py completed successfully ✅")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "Task completed successfully!"
                            })
                
                except subprocess.CalledProcessError as e:
                    logging.error(f"Error running widgets.py: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error running widgets.py: {e}"
                    })
                
                return {
                    "status": "plan_provided",
                    "question": "",
                    "steps": result["steps"],
                    "task_description": result["task_description"]
                }
            else:
                logging.warning("No .txt files found in steps/ folder")
                output = "No task plan files found in steps/ folder"
                logging.info(f"Task info extracted: {output}")
                st.session_state.messages.append({"role": "assistant", "content": output})
                
                # Run widgets.py with the error output as input argument
                input_argument = output
                try:
                    subprocess.run(["python", script_path, input_argument], check=True)
                    logging.info(f"Successfully ran widgets.py with input: {input_argument}")
                    
                    # Run manage.py with no timestamp (fallback)
                    with st.spinner("Thinking <running manage.py file...>"):
                        manage_result = run_manage_py("")
                        if manage_result["status"] == "error":
                            logging.error(f"manage.py failed: {manage_result['message']}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Error running task: {manage_result['message']}"
                            })
                        else:
                            logging.info("manage.py completed successfully ✅")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "Task completed successfully!"
                            })
                
                except subprocess.CalledProcessError as e:
                    logging.error(f"Error running widgets.py: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error running widgets.py: {e}"
                    })
                
                return {
                    "status": "plan_provided",
                    "question": "",
                    "steps": result["steps"],
                    "task_description": result["task_description"]
                }
        elif result["status"] == "conversational":
            if is_task:
                st.session_state.task_context = reset_task_context()
                task_context = st.session_state.task_context
                task_context["task"] = prompt
                task_context["item"] = order_pattern.group(2).strip() if order_pattern else None
                question_type, question = get_next_question_type(task_context)
                task_context["questions_asked"].append(question)
                if question_type != "none":
                    task_context["question_types_asked"].append(question_type)
                task_context["question_count"] += 1
                st.session_state.task_context = task_context
                return {
                    "status": "clarification_needed",
                    "question": question,
                    "steps": [],
                    "task_description": ""
                }
        if task_context["confirmation_asked"]:
            if is_confirmation_response or task_context.get("confirmation_count", 0) >= 2:
                specs = task_context["details"].get("specs", "")
                source = task_context["details"].get("source", "the specified service")
                source_url = f"https://www.{source}.com" if source in common_providers else f"https://www.{source}.com"
                product_name = clean_product_name(task_context["item"])
                steps = [
                    f"Open a web browser like Google Chrome or Firefox.",
                    f"Navigate to {source}'s website at {source_url}.",
                    f"Prepare to search for the product: {product_name}.",
                    f"Locate the search bar, usually at the top of the {source} homepage.",
                    f"Click the search bar and type '{product_name}'.",
                    f"Press Enter or click the search icon to start the search.",
                    f"Wait for the search results to load completely.",
                    f"If available, apply filters based on your specifications: {specs}.",
                    f"Wait for the page to reload after applying filters.",
                    f"Sort the results by relevance or customer reviews if available.",
                    f"Look through the search results to find a {product_name} that matches your specifications: {specs}.",
                    f"Select a {product_name} by clicking its title or image to visit its product page.",
                    f"Verify the product details on the page to confirm they match your specifications: {specs}.",
                    f"Check the price to ensure it fits within your budget: {task_context['details'].get('budget', 'no budget specified')}.",
                    f"Locate the 'Add to Cart' button on the product page.",
                    f"Click 'Add to Cart' to add the {product_name} to your cart.",
                    f"Wait for a confirmation that the {product_name} was added to your cart.",
                    f"If a warranty or additional offer pop-up appears, select 'No Thanks' or decline.",
                    f"If a CAPTCHA appears, follow the on-screen instructions to complete it.",
                    f"Click the cart icon, usually in the top-right corner, to view your cart.",
                    f"Verify that the {product_name} in the cart matches your specifications: {specs}.",
                    f"If the item is incorrect, click 'Remove' to delete it from the cart.",
                    f"Click 'Proceed to Checkout' to start the checkout process.",
                    f"If prompted to sign in, enter your email and password to log in.",
                    f"Select the delivery option '{task_context['details'].get('delivery', 'standard')}'.",
                    f"Enter or confirm your delivery address.",
                    f"Review the order summary to ensure the {product_name} and total cost are correct.",
                    f"Click 'Place Order' or 'Confirm Order' to complete the purchase.",
                    f"Wait for the order confirmation page and note the order number or take a screenshot.",
                    f"Close the browser."
                ]
                budget_info = task_context['details'].get('budget', 'no budget specified')
                task_description = (
                    f"The task is to order a {product_name} from {source} "
                    f"with the following specifications: {specs or 'no specific preferences provided'}. "
                    f"The order will be completed with {task_context['details'].get('delivery', 'standard')} delivery, "
                    f"and the budget is {budget_info}. "
                    f"The order is to be placed as soon as possible unless otherwise specified."
                )
                filename = save_to_file(steps, task_description)
                task_context["details"]["steps_provided"] = True
                st.session_state.task_context = reset_task_context()
                
                time.sleep(1)
                steps_folder = "steps"
                txt_files = glob.glob(os.path.join(steps_folder, "*.txt"))
                if txt_files:
                    txt_files.sort(key=os.path.getctime, reverse=True)
                    first_txt_file = txt_files[0]
                    logging.info(f"Selected first .txt file: {first_txt_file}")
                    
                    task_timestamp = extract_timestamp_from_filename(first_txt_file)
                    logging.info(f"Extracted task timestamp: {task_timestamp}")
                    
                    output = process_task_plan_file(first_txt_file, api_key, task_context.get("item", ""))
                    logging.info(f"Task info extracted: {output}")
                    st.session_state.messages.append({"role": "assistant", "content": output})
                    
                    # Run widgets.py with the LLM output as input argument
                    input_argument = output
                    try:
                        subprocess.run(["python", script_path, input_argument], check=True)
                        logging.info(f"Successfully ran widgets.py with input: {input_argument}")
                        
                        # Run manage.py with the task timestamp
                        with st.spinner("Thinking <running manage.py file...>"):
                            manage_result = run_manage_py(task_timestamp)
                            if manage_result["status"] == "error":
                                logging.error(f"manage.py failed: {manage_result['message']}")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"Error running task: {manage_result['message']}"
                                })
                            else:
                                logging.info("manage.py completed successfully ✅")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": "Task completed successfully!"
                                })
                    
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Error running widgets.py: {e}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error running widgets.py: {e}"
                        })
                    
                    return {
                        "status": "plan_provided",
                        "question": "",
                        "steps": steps,
                        "task_description": task_description
                    }
                else:
                    logging.warning("No .txt files found in steps/ folder")
                    output = "No task plan files found in steps/ folder"
                    logging.info(f"Task info extracted: {output}")
                    st.session_state.messages.append({"role": "assistant", "content": output})
                    
                    # Run widgets.py with the error output as input argument
                    input_argument = output
                    try:
                        subprocess.run(["python", script_path, input_argument], check=True)
                        logging.info(f"Successfully ran widgets.py with input: {input_argument}")
                        
                        # Run manage.py with no timestamp (fallback)
                        with st.spinner("Thinking <running manage.py file...>"):
                            manage_result = run_manage_py("")
                            if manage_result["status"] == "error":
                                logging.error(f"manage.py failed: {manage_result['message']}")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"Error running task: {manage_result['message']}"
                                })
                            else:
                                logging.info("manage.py completed successfully ✅")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": "Task completed successfully!"
                                })
                    
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Error running widgets.py: {e}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error running widgets.py: {e}"
                        })
                    
                    return {
                        "status": "plan_provided",
                        "question": "",
                        "steps": steps,
                        "task_description": task_description
                    }
        st.session_state.task_context = task_context
        logging.info(f"Updated Task Context: {json.dumps(task_context)}")
        
        return result
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse AI Agent response: {e}, Raw response: {raw_response}, Extracted JSON: {json_str if 'json_str' in locals() else 'None'}")
        if not task_context["task"]:
            task_context["task"] = prompt
            task_context["item"] = order_pattern.group(2).strip() if order_pattern else None
            question_type, question = get_next_question_type(task_context)
            task_context["questions_asked"].append(question)
            if question_type != "none":
                task_context["question_types_asked"].append(question_type)
            task_context["question_count"] += 1
            st.session_state.task_context = task_context
            return {
                "status": "clarification_needed",
                "question": question,
                "steps": [],
                "task_description": ""
            }
        if is_source_response:
            task_context["details"]["source"] = prompt.lower()
            question_type, question = get_next_question_type(task_context)
            task_context["questions_asked"].append(question)
            if question_type != "none":
                task_context["question_types_asked"].append(question_type)
            task_context["question_count"] += 1
            st.session_state.task_context = task_context
            return {
                "status": "clarification_needed",
                "question": question,
                "steps": [],
                "task_description": ""
            }
        if task_context["confirmation_asked"]:
            if is_confirmation_response:
                specs = task_context["details"].get("specs", "")
                source = task_context["details"].get("source", "the specified service")
                source_url = f"https://www.{source}.com" if source in common_providers else f"https://www.{source}.com"
                product_name = clean_product_name(task_context["item"])
                steps = [
                    f"Open a web browser like Google Chrome or Firefox.",
                    f"Navigate to {source}'s website at {source_url}.",
                    f"Prepare to search for the product: {product_name}.",
                    f"Locate the search bar, usually at the top of the {source} homepage.",
                    f"Click the search bar and type '{product_name}'.",
                    f"Press Enter or click the search icon to start the search.",
                    f"Wait for the search results to load completely.",
                    f"If available, apply filters based on your specifications: {specs}.",
                    f"Wait for the page to reload after applying filters.",
                    f"Sort the results by relevance or customer reviews if available.",
                    f"Look through the search results to find a {product_name} that matches your specifications: {specs}.",
                    f"Select a {product_name} by clicking its title or image to visit its product page.",
                    f"Verify the product details on the page to confirm they match your specifications: {specs}.",
                    f"Check the price to ensure it fits within your budget: {task_context['details'].get('budget', 'no budget specified')}.",
                    f"Locate the 'Add to Cart' button on the product page.",
                    f"Click 'Add to Cart' to add the {product_name} to your cart.",
                    f"Wait for a confirmation that the {product_name} was added to your cart.",
                    f"If a warranty or additional offer pop-up appears, select 'No Thanks' or decline.",
                    f"If a CAPTCHA appears, follow the on-screen instructions to complete it.",
                    f"Click the cart icon, usually in the top-right corner, to view your cart.",
                    f"Verify that the {product_name} in the cart matches your specifications: {specs}.",
                    f"If the item is incorrect, click 'Remove' to delete it from the cart.",
                    f"Click 'Proceed to Checkout' to start the checkout process.",
                    f"If prompted to sign in, enter your email and password to log in.",
                    f"Select the delivery option '{task_context['details'].get('delivery', 'standard')}'.",
                    f"Enter or confirm your delivery address.",
                    f"Review the order summary to ensure the {product_name} and total cost are correct.",
                    f"Click 'Place Order' or 'Confirm Order' to complete the purchase.",
                    f"Wait for the order confirmation page and note the order number or take a screenshot.",
                    f"Close the browser."
                ]
                budget_info = task_context['details'].get('budget', 'no budget specified')
                task_description = (
                    f"The task is to order a {product_name} from {source} "
                    f"with the following specifications: {specs or 'no specific preferences provided'}. "
                    f"The order will be completed with {task_context['details'].get('delivery', 'standard')} delivery, "
                    f"and the budget is {budget_info}. "
                    f"The order is to be placed as soon as possible unless otherwise specified."
                )
                filename = save_to_file(steps, task_description)
                task_context["details"]["steps_provided"] = True
                st.session_state.task_context = reset_task_context()
                
                time.sleep(1)
                steps_folder = "steps"
                txt_files = glob.glob(os.path.join(steps_folder, "*.txt"))
                if txt_files:
                    txt_files.sort(key=os.path.getctime, reverse=True)
                    first_txt_file = txt_files[0]
                    logging.info(f"Selected first .txt file: {first_txt_file}")
                    
                    task_timestamp = extract_timestamp_from_filename(first_txt_file)
                    logging.info(f"Extracted task timestamp: {task_timestamp}")
                    
                    output = process_task_plan_file(first_txt_file, api_key, task_context.get("item", ""))
                    logging.info(f"Task info extracted: {output}")
                    st.session_state.messages.append({"role": "assistant", "content": output})
                    
                    # Run widgets.py with the LLM output as input argument
                    input_argument = output
                    try:
                        subprocess.run(["python", script_path, input_argument], check=True)
                        logging.info(f"Successfully ran widgets.py with input: {input_argument}")
                        
                        # Run manage.py with the task timestamp
                        with st.spinner("Thinking <running manage.py file...>"):
                            manage_result = run_manage_py(task_timestamp)
                            if manage_result["status"] == "error":
                                logging.error(f"manage.py failed: {manage_result['message']}")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"Error running task: {manage_result['message']}"
                                })
                            else:
                                logging.info("manage.py completed successfully ✅")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": "Task completed successfully!"
                                })
                    
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Error running widgets.py: {e}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error running widgets.py: {e}"
                        })
                    
                    return {
                        "status": "plan_provided",
                        "question": "",
                        "steps": steps,
                        "task_description": task_description
                    }
                else:
                    logging.warning("No .txt files found in steps/ folder")
                    output = "No task plan files found in steps/ folder"
                    logging.info(f"Task info extracted: {output}")
                    st.session_state.messages.append({"role": "assistant", "content": output})
                    
                    # Run widgets.py with the error output as input argument
                    input_argument = output
                    try:
                        subprocess.run(["python", script_path, input_argument], check=True)
                        logging.info(f"Successfully ran widgets.py with input: {input_argument}")
                        
                        # Run manage.py with no timestamp (fallback)
                        with st.spinner("Thinking <running manage.py file...>"):
                            manage_result = run_manage_py("")
                            if manage_result["status"] == "error":
                                logging.error(f"manage.py failed: {manage_result['message']}")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"Error running task: {manage_result['message']}"
                                })
                            else:
                                logging.info("manage.py completed successfully ✅")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": "Task completed successfully!"
                                })
                    
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Error running widgets.py: {e}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error running widgets.py: {e}"
                        })
                    
                    return {
                        "status": "plan_provided",
                        "question": "",
                        "steps": steps,
                        "task_description": task_description
                    }
            else:
                if prompt.lower() not in ["no", "nothing", "no nothing", "nothing else", "looks good"]:
                    task_context["details"]["specs"] = f"{task_context['details'].get('specs', '')}, {prompt.lower()}".strip(", ")
                    task_context["confirmation_asked"] = True
                    task_context["confirmation_count"] = task_context.get("confirmation_count", 0) + 1
                    st.session_state.task_context = task_context
                    return {
                        "status": "confirmation_needed",
                        "question": "Is there anything else you’d like to add before I create the plan?",
                        "steps": [],
                        "task_description": ""
                    }
        question_type, question = get_next_question_type(task_context)
        task_context["questions_asked"].append(question)
        if question_type != "none":
            task_context["question_types_asked"].append(question_type)
        task_context["question_count"] += 1
        st.session_state.task_context = task_context
        return {
            "status": "clarification_needed",
            "question": question,
            "steps": [],
            "task_description": ""
        }
    except Exception as e:
        logging.error(f"Unexpected error in AI Agent processing: {e}")
        if not task_context["task"]:
            task_context["task"] = prompt
            task_context["item"] = order_pattern.group(2).strip() if order_pattern else None
            question_type, question = get_next_question_type(task_context)
            task_context["questions_asked"].append(question)
            if question_type != "none":
                task_context["question_types_asked"].append(question_type)
            task_context["question_count"] += 1
            st.session_state.task_context = task_context
            return {
                "status": "clarification_needed",
                "question": question,
                "steps": [],
                "task_description": ""
            }
        if is_source_response:
            task_context["details"]["source"] = prompt.lower()
            question_type, question = get_next_question_type(task_context)
            task_context["questions_asked"].append(question)
            if question_type != "none":
                task_context["question_types_asked"].append(question_type)
            task_context["question_count"] += 1
            st.session_state.task_context = task_context
            return {
                "status": "clarification_needed",
                "question": question,
                "steps": [],
                "task_description": ""
            }
        if task_context["confirmation_asked"]:
            if is_confirmation_response:
                specs = task_context["details"].get("specs", "")
                source = task_context["details"].get("source", "the specified service")
                source_url = f"https://www.{source}.com" if source in common_providers else f"https://www.{source}.com"
                product_name = clean_product_name(task_context["item"])
                steps = [
                    f"Open a web browser like Google Chrome or Firefox.",
                    f"Navigate to {source}'s website at {source_url}.",
                    f"Prepare to search for the product: {product_name}.",
                    f"Locate the search bar, usually at the top of the {source} homepage.",
                    f"Click the search bar and type '{product_name}'.",
                    f"Press Enter or click the search icon to start the search.",
                    f"Wait for the search results to load completely.",
                    f"If available, apply filters based on your specifications: {specs}.",
                    f"Wait for the page to reload after applying filters.",
                    f"Sort the results by relevance or customer reviews if available.",
                    f"Look through the search results to find a {product_name} that matches your specifications: {specs}.",
                    f"Select a {product_name} by clicking its title or image to visit its product page.",
                    f"Verify the product details on the page to confirm they match your specifications: {specs}.",
                    f"Check the price to ensure it fits within your budget: {task_context['details'].get('budget', 'no budget specified')}.",
                    f"Locate the 'Add to Cart' button on the product page.",
                    f"Click 'Add to Cart' to add the {product_name} to your cart.",
                    f"Wait for a confirmation that the {product_name} was added to your cart.",
                    f"If a warranty or additional offer pop-up appears, select 'No Thanks' or decline.",
                    f"If a CAPTCHA appears, follow the on-screen instructions to complete it.",
                    f"Click the cart icon, usually in the top-right corner, to view your cart.",
                    f"Verify that the {product_name} in the cart matches your specifications: {specs}.",
                    f"If the item is incorrect, click 'Remove' to delete it from the cart.",
                    f"Click 'Proceed to Checkout' to start the checkout process.",
                    f"If prompted to sign in, enter your email and password to log in.",
                    f"Select the delivery option '{task_context['details'].get('delivery', 'standard')}'.",
                    f"Enter or confirm your delivery address.",
                    f"Review the order summary to ensure the {product_name} and total cost are correct.",
                    f"Click 'Place Order' or 'Confirm Order' to complete the purchase.",
                    f"Wait for the order confirmation page and note the order number or take a screenshot.",
                    f"Close the browser."
                ]
                budget_info = task_context['details'].get('budget', 'no budget specified')
                task_description = (
                    f"The task is to order a {product_name} from {source} "
                    f"with the following specifications: {specs or 'no specific preferences provided'}. "
                    f"The order will be completed with {task_context['details'].get('delivery', 'standard')} delivery, "
                    f"and the budget is {budget_info}. "
                    f"The order is to be placed as soon as possible unless otherwise specified."
                )
                filename = save_to_file(steps, task_description)
                task_context["details"]["steps_provided"] = True
                st.session_state.task_context = reset_task_context()
                
                time.sleep(1)
                steps_folder = "steps"
                txt_files = glob.glob(os.path.join(steps_folder, "*.txt"))
                if txt_files:
                    txt_files.sort(key=os.path.getctime, reverse=True)
                    first_txt_file = txt_files[0]
                    logging.info(f"Selected first .txt file: {first_txt_file}")
                    
                    task_timestamp = extract_timestamp_from_filename(first_txt_file)
                    logging.info(f"Extracted task timestamp: {task_timestamp}")
                    
                    output = process_task_plan_file(first_txt_file, api_key, task_context.get("item", ""))
                    logging.info(f"Task info extracted: {output}")
                    st.session_state.messages.append({"role": "assistant", "content": output})
                    
                    # Run widgets.py with the LLM output as input argument
                    input_argument = output
                    try:
                        subprocess.run(["python", script_path, input_argument], check=True)
                        logging.info(f"Successfully ran widgets.py with input: {input_argument}")
                        
                        # Run manage.py with the task timestamp
                        with st.spinner("Thinking <running manage.py file...>"):
                            manage_result = run_manage_py(task_timestamp)
                            if manage_result["status"] == "error":
                                logging.error(f"manage.py failed: {manage_result['message']}")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"Error running task: {manage_result['message']}"
                                })
                            else:
                                logging.info("manage.py completed successfully ✅")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": "Task completed successfully!"
                                })
                    
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Error running widgets.py: {e}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error running widgets.py: {e}"
                        })
                    
                    return {
                        "status": "plan_provided",
                        "question": "",
                        "steps": steps,
                        "task_description": task_description
                    }
                else:
                    logging.warning("No .txt files found in steps/ folder")
                    output = "No task plan files found in steps/ folder"
                    logging.info(f"Task info extracted: {output}")
                    st.session_state.messages.append({"role": "assistant", "content": output})
                    
                    # Run widgets.py with the error output as input argument
                    input_argument = output
                    try:
                        subprocess.run(["python", script_path, input_argument], check=True)
                        logging.info(f"Successfully ran widgets.py with input: {input_argument}")
                        
                        # Run manage.py with no timestamp (fallback)
                        with st.spinner("Thinking <running manage.py file...>"):
                            manage_result = run_manage_py("")
                            if manage_result["status"] == "error":
                                logging.error(f"manage.py failed: {manage_result['message']}")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"Error running task: {manage_result['message']}"
                                })
                            else:
                                logging.info("manage.py completed successfully ✅")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": "Task completed successfully!"
                                })
                    
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Error running widgets.py: {e}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error running widgets.py: {e}"
                        })
                    
                    return {
                        "status": "plan_provided",
                        "question": "",
                        "steps": steps,
                        "task_description": task_description
                    }
            else:
                if prompt.lower() not in ["no", "nothing", "no nothing", "nothing else", "looks good"]:
                    task_context["details"]["specs"] = f"{task_context['details'].get('specs', '')}, {prompt.lower()}".strip(", ")
                    task_context["confirmation_asked"] = True
                    task_context["confirmation_count"] = task_context.get("confirmation_count", 0) + 1
                    st.session_state.task_context = task_context
                    return {
                        "status": "confirmation_needed",
                        "question": "Is there anything else you’d like to add before I create the plan?",
                        "steps": [],
                        "task_description": ""
                    }
        question_type, question = get_next_question_type(task_context)
        task_context["questions_asked"].append(question)
        if question_type != "none":
            task_context["question_types_asked"].append(question_type)
        task_context["question_count"] += 1
        st.session_state.task_context = task_context
        return {
            "status": "clarification_needed",
            "question": question,
            "steps": [],
            "task_description": ""
        }

if api_key:
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=3500)
    arxiv = ArxivQueryRun(name="arxiv", api_wrapper=arxiv_wrapper)
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3500)
    wiki = WikipediaQueryRun(name="wikipedia", api_wrapper=wiki_wrapper)
    search = DuckDuckGoSearchRun(name="Search")
    tools = [search, arxiv, wiki]

    if enable_ai_agent:
        llm_model = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-70b-8192",
            streaming=True,
            max_tokens=1000
        )
        ai_agent_system_prompt = (
            "You are NeuraBot-AI-Agent, a friendly and polite assistant that helps users by performing tasks with detailed, human-readable step-by-step plans and engaging in natural, human-like conversation. "
            "Your goal is to assist with tasks (e.g., ordering items, planning events) while responding courteously to all inputs, including casual or conversational ones (e.g., 'Thanks,' 'Great'). "
            "The task flow must be generic and work for any service provider or task type without hardcoding specific websites, products, or details. "
            "Return ONLY valid JSON, with no additional text outside the JSON object. "
            "Follow these steps for every user input:\n"
            "1. **Classify Input**: Determine if the input is a task, conversational, or a confirmation response.\n"
            "   - **Task Inputs**: Contain action-oriented words (e.g., 'order,' 'buy,' 'purchase,' 'plan,' 'book,' 'get,' 'make') followed by an item or goal. Examples: 'Order a shoe,' 'Buy a laptop,' 'Plan a trip.'\n"
            "   - **Conversational Inputs**: Include gratitude, feedback, greetings, or vague phrases (e.g., 'Thanks,' 'Cool thanks,' 'Looks good'), but exclude 'no,' 'nothing,' 'no nothing,' 'nothing else' when part of a confirmation response.\n"
            "   - **Confirmation Responses**: Inputs like 'no,' 'nothing,' 'no nothing,' 'nothing else,' or 'looks good' when a confirmation question has been asked.\n"
            "   - If the input contains task-related terms and no task exists or the previous task is complete, treat it as a new task and reset the task context.\n"
            "2. **Handle Conversational Inputs**: If the input is conversational and not a confirmation response or task:\n"
            "   - Respond politely (e.g., for 'Thanks,' reply 'You’re very welcome! Anything else I can help with?').\n"
            "   - If no active task, reset the task context.\n"
            "   - If a task is active, preserve the task context unless the previous task is complete.\n"
            "   - Return JSON with status 'conversational' and a friendly message in 'question'.\n"
            "3. **Handle Task Inputs**: If the input is a task or part of an ongoing task:\n"
            "   - **Identify Task**: Determine the task and item from the input (e.g., 'Order a laptop' → task: order, item: laptop). Use history to resolve vague terms.\n"
            "   - **Reset Context for New Task**: If the previous task is complete (steps provided) or no task exists, reset the task context before initializing the new task.\n"
            "   - **Ask Clarification**: If details are missing, dynamically generate ONE relevant, grammatically correct, and task-specific clarification question (up to 5-7 total). Examples of question types:\n"
            "     - Source: Ask where the task should be performed if not specified.\n"
            "     - Specifications: Ask about desired features or preferences for the item.\n"
            "     - Budget: Ask about the user’s budget or price range.\n"
            "     - Delivery: Ask about delivery or pickup preferences.\n"
            "     - Timing: Ask about the desired timing for the task.\n"
            "     Ensure questions are polite, clear, and tailored to the task without hardcoded examples. Use the item name and context to make questions specific. Avoid repeating question types.\n"
            "   - **Handle Source Responses**: If the input is a service provider name or a short phrase indicating a provider, store it as the source and ask a follow-up question about item specifications.\n"
            "   - **Track Details**: Store details in context (e.g., {'source': 'Amazon', 'item': 'laptop', 'specs': '64GB RAM'}). Do not repeat known details. Exclude confirmation responses from specs.\n"
            "   - **Check Sufficiency**: Continue asking questions until critical details are gathered (e.g., source, specs, budget, delivery).\n"
            "   - **Confirm**: When details are sufficient, ask: 'Is there anything else you’d like to add before I create the plan?' only once unless new specs are added.\n"
            "   - **Handle Confirmation Response**:\n"
            "     - If the response adds task-related details, append to specs and re-ask the confirmation question.\n"
            "     - If the response is 'no,' 'nothing,' 'no nothing,' 'nothing else,' or 'looks good,' generate the plan immediately.\n"
            "     - If unclear but task-related, ask a specific follow-up.\n"
            "     - Ignore confirmation responses in specs.\n"
            "     - If the confirmation question has been asked twice without new specs, generate the plan automatically.\n"
            "   - **Generate Plan**: Only after confirmation with no additional details, provide a plan with 20-29 human-readable steps that cover essential actions (e.g., navigating to the website, searching for the product, applying filters, adding to cart, checking out). Avoid technical terms and overly granular steps. The first step must navigate to the base URL of the user-specified service provider (e.g., 'https://www.amazon.com'). The second step must specify the product name to be searched (e.g., 'sports shoes'). Do not include a search query in the URL (e.g., avoid '/s?k=...'). Use the user-specified service provider and embed all user-provided details in the steps, stopping before payment. For single-item tasks, focus on selecting one item with the specified details.\n"
            "   - **Task Description**: Provide a clear summary starting with 'The task is to...', including the item, source, specifications, delivery method, budget, and timing.\n"
            "4. **Response Format**: Return ONLY valid JSON:\n"
            "   ```json\n"
            "   {\n"
            "     \"status\": \"conversational\" or \"clarification_needed\" or \"confirmation_needed\" or \"plan_provided\",\n"
            "     \"question\": \"conversational response, clarification question, confirmation, or empty string\",\n"
            "     \"steps\": [\"step 1\", \"step 2\", ...] if plan_provided, else [],\n"
            "     \"task_description\": \"summary if plan_provided, else empty string\"\n"
            "   }\n"
            "   ```\n"
            "Rules:\n"
            "- Use a polite, friendly tone in all responses.\n"
            "- For conversational inputs, maintain or reset task context based on task completion, but do not treat confirmation responses as conversational.\n"
            "- Reset task context for new tasks after a task is complete or if no task is active.\n"
            "- Ask 5-7 relevant questions for tasks unless details are sufficient, generating questions dynamically based on the task and item.\n"
            "- Ensure questions are grammatically correct, task-specific, and include the item name (e.g., 'What features are you looking for in your laptop?').\n"
            "- Avoid repeating question types (e.g., two source-related questions) by tracking question types.\n"
            "- Ask confirmation only once unless new specs are provided, and generate the plan after a second confirmation attempt if no new specs are added.\n"
            "- Generate steps only after the user confirms no additional details or after two confirmation prompts.\n"
            "- Use the user-specified service provider in steps and task description (e.g., 'Amazon' instead of '[service’s website]').\n"
            "- The first step must use the base URL (e.g., 'https://www.amazon.com') without any search query. The second step must specify the product name for searching.\n"
            "- Save the task plan and description to a .txt file in a 'steps' folder with a timestamp-based filename.\n"
            "- After providing a plan, reset the task context to prevent further clarification prompts.\n"
            "- If input is vague but task-related, ask a specific follow-up.\n"
            "- For errors or non-task inputs, return a clarification question.\n"
        )
    else:
        llm_model = ChatGroq(
            groq_api_key=api_key,
            model_name="Llama-3.1-8b-Instant",
            streaming=True,
            max_tokens=2500
        )
        ai_agent_system_prompt = None

    if prompt := st.chat_input(placeholder="Chat with me about anything. Built by Sujal"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            if enable_ai_agent:
                response = process_ai_agent_input(prompt, llm_model, ai_agent_system_prompt, api_key)
                if response["status"] in ["clarification_needed", "confirmation_needed"]:
                    if not st.session_state.task_context["task"]:
                        st.session_state.task_context["task"] = prompt
                    st.write(response["question"])
                    st.session_state.messages.append({"role": "assistant", "content": response["question"]})
                elif response["status"] == "plan_provided":
                    # No output displayed in Streamlit UI
                    pass
                elif response["status"] == "conversational":
                    st.write(response["question"])
                    st.session_state.messages.append({"role": "assistant", "content": response["question"]})
                else:
                    error_msg = response["task_description"] or "I’m sorry, I couldn’t process that request. Could you clarify what you’d like to do?"
                    st.session_state.task_context = reset_task_context()
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.write(error_msg)
            else:
                final_response = process_default_input(prompt, llm_model, tools)
                st.session_state.task_context = reset_task_context()
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                st.write(final_response)

    if (st.session_state.task_context["task"] and enable_ai_agent and 
        (st.session_state.task_context["question_count"] > 0 or st.session_state.task_context["confirmation_asked"]) and 
        not st.session_state.task_context["details"].get("steps_provided")):
        with st.form(key="clarification_form"):
            clarification_answer = st.text_input(
                f"Answer: {st.session_state.task_context['questions_asked'][-1] if st.session_state.task_context['questions_asked'] and not st.session_state.task_context['confirmation_asked'] else 'Is there anything else you’d like to add before I create the plan?'}",
                key="clarification_input"
            )
            if st.form_submit_button("Submit"):
                if clarification_answer.strip():
                    st.session_state.messages.append({"role": "user", "content": clarification_answer})
                    with st.chat_message("user"):
                        st.write(clarification_answer)
                    with st.chat_message("assistant"):
                        details = st.session_state.task_context["details"]
                        question_asked = st.session_state.task_context["questions_asked"][-1] if st.session_state.task_context["questions_asked"] else ""
                        if "where" in question_asked.lower() or "service" in question_asked.lower():
                            details["source"] = clarification_answer.lower()
                        elif "features" in question_asked.lower() or "specifications" in question_asked.lower():
                            details["specs"] = clarification_answer.lower()
                        elif "budget" in question_asked.lower():
                            details["budget"] = clarification_answer.lower()
                        elif "delivery" in question_asked.lower() or "pickup" in question_asked.lower():
                            details["delivery"] = clarification_answer.lower()
                        elif "when" in question_asked.lower() or "timing" in question_asked.lower():
                            details["timing"] = clarification_answer.lower()
                        st.session_state.task_context["details"] = details
                        logging.info(f"Updated Task Context: {json.dumps(st.session_state.task_context)}")
                        
                        response = process_ai_agent_input(clarification_answer, llm_model, ai_agent_system_prompt, api_key)
                        if response["status"] in ["clarification_needed", "confirmation_needed"]:
                            st.write(response["question"])
                            st.session_state.messages.append({"role": "assistant", "content": response["question"]})
                        elif response["status"] == "plan_provided":
                            # No output displayed in Streamlit UI
                            pass
                        elif response["status"] == "conversational":
                            st.write(response["question"])
                            st.session_state.messages.append({"role": "assistant", "content": response["question"]})
                        else:
                            error_msg = response["task_description"] or "I’m sorry, I couldn’t process that request. Could you clarify what you’d like to do?"
                            st.session_state.task_context = reset_task_context()
                            st.write(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    st.error("Please provide a valid answer.")
else:
    st.warning("Please enter your GROQ API key in the sidebar to proceed.")