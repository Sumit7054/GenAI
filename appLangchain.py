import autogen
import os
from pathlib import Path
from autogen.coding import LocalCommandLineCodeExecutor

# To use this script, you need to have Ollama installed and running.
# Install Ollama from https://ollama.com/
# Then, pull the model: ollama pull qwen2.5-coder:7b
# Start Ollama server: ollama serve
# Make sure it's running on http://localhost:11434/v1

# Requirements:
# - Python 3.10.0
# - pyautogen: pip install pyautogen~=0.2

# Constants
BASE_URL = "http://localhost:11434/v1"
MODEL = "deepseek-r1"
WORK_DIR = Path("./data_science_pipeline")

# Create work directory if not exists
WORK_DIR.mkdir(exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir=WORK_DIR)


# Ollama configuration
config_list = [{"base_url": BASE_URL, 'api_type':'ollama', 'api_key': "fakekey", 'model': MODEL}]
llm_config = {"config_list": config_list}

# Define agents
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"executor": code_executor},
    is_termination_msg=lambda msg: "FINISH" in msg.get("content"),
)

planner = autogen.AssistantAgent(
    name="planner",
    system_message="""You are a planner agent. Based on the user's requirements, plan the steps for a data science pipeline. List the steps in order, such as:
1. Load data from [path]
2. Perform exploratory data analysis (EDA)
3. Clean and preprocess the data
4. Feature engineering
5. Train machine learning models (e.g., Logistic Regression, Random Forest, etc.)
6. Hyperparameter tuning
7. Evaluate model performance
8. Model explainability
Make sure to include all necessary steps based on the requirements.""",
    llm_config=llm_config,
)

code_writer = autogen.AssistantAgent(
    name="code_writer",
    system_message="""You are a code writer agent. Write Python code for the specified step in the data science pipeline. Include necessary library installations using subprocess.run(['pip', 'install', 'library_name']). Save any outputs or plots to files in the working directory. Make sure the code is complete and executable.""",
    llm_config=llm_config,
)

thinker = autogen.AssistantAgent(
    name="thinker",
    system_message="""You are a thinker agent. Provide advice and decision-making for the data science pipeline. When asked, suggest methods, parameters, or approaches to improve the results or fix issues. Use first principles and analogies to reason about the problems.""",
    llm_config=llm_config,
)

checker = autogen.AssistantAgent(
    name="checker",
    system_message="""You are a checker agent. Verify the output of each step in the pipeline. Check if the step was performed correctly and if there are any issues. If everything is okay, say 'OK'. If there are issues, describe them and ask for advice from the thinker.""",
    llm_config=llm_config,
)

# Set up group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, code_writer, thinker, checker],
    messages=[],
    max_round=50,
    speaker_selection_method="auto",
)

# Manager system message
manager_system_message = """You are the manager of a data science pipeline team. Your task is to guide the conversation through the following process:

1. Start by having the planner plan the pipeline steps based on the user's requirements provided by the user_proxy.

2. The planner should list the steps in order, such as '1. Load data from [path]', '2. Perform EDA', etc.

3. For each step, ask the code_writer to write the Python code necessary to perform that step. The code should be complete and executable, including any necessary library installations.

4. After the code_writer provides the code, have the user_proxy execute the code. The user_proxy will send the output of the code execution.

5. Then, have the checker verify the output. The checker should check if the step was performed correctly and if there are any issues. If everything is okay, proceed to the next step. If there are issues, ask the thinker for advice on how to fix them.

6. The thinker can provide suggestions or decisions on methods, parameters, or approaches to improve the results.

7. Based on the thinker's advice, have the code_writer rewrite the code for that step and repeat the execution and checking process until the step is completed successfully.

8. Continue this process for all steps until the entire pipeline is completed.

9. Once all steps are done, have the user_proxy summarize the results and present them to the user.

Make sure to keep track of the current step and ensure that each step is completed before moving to the next one. Also, handle any errors or issues that arise during code execution or verification."""

manager_llm_config = {
    "config_list": config_list,
}

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=manager_llm_config,
)

# Get user input
dataset_path = input("Enter the path to the dataset: ")
requirements = input("Enter the specific requirements: ")

initial_message = f"Plan a data science pipeline using the dataset at {dataset_path} with the following requirements: {requirements}"

# Initiate chat
user_proxy.initiate_chat(manager, message=initial_message)