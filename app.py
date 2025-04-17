import streamlit as st
import pandas as pd
from pandasai import Agent
from pandasai.llm import HuggingFaceTextGen
import matplotlib.pyplot as plt
from io import StringIO
import sys
import traceback

# Configuration
LLM_API_KEY = ""  
LLM_ENDPOINT = "https://api-inference.huggingface.co/models/gpt2"

# Initialize LLM with compatibility checks
try:
    llm = HuggingFaceTextGen(
        api_key=LLM_API_KEY,
        endpoint=LLM_ENDPOINT,
        max_new_tokens=1000,
        temperature=0.1
    )
except Exception as e:
    st.error(f"LLM initialization failed: {str(e)}")
    st.stop()

# Initialize PandasAI Agent with error recovery
def get_agent(df):
    return Agent(
        df,
        config={
            "llm": llm,
            "save_logs": True,
            "verbose": True,
            "enable_cache": False
        }
    )

# State management
STATE = {
    "current_step": 1,
    "data_versions": [],
    "model": None,
    "target": None,
    "error_info": None
}

def reset_state():
    STATE.update({
        "current_step": 1,
        "data_versions": [],
        "model": None,
        "target": None,
        "error_info": None
    })

# UI Components
def show_data_characteristics(df):
    with st.expander("📊 Complete Data Profile", expanded=True):
        cols = st.columns(3)
        cols[0].metric("Rows", df.shape[0])
        cols[1].metric("Columns", df.shape[1])
        cols[2].metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("Type Distribution")
        type_counts = df.dtypes.value_counts().reset_index()
        type_counts.columns = ["Data Type", "Count"]
        st.bar_chart(type_counts.set_index("Data Type"))
        
        st.subheader("Sample Statistics")
        st.write(df.describe(include='all'))

# Error handling system
class WorkflowGuard:
    def __init__(self, step):
        self.step = step
        
    def __enter__(self):
        pass
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            STATE["error_info"] = {
                "step": self.step,
                "type": exc_type.__name__,
                "message": str(exc_val),
                "traceback": traceback.format_exc()
            }
            st.error(f"🚨 Step {self.step} Failed: {str(exc_val)}")
            show_troubleshooting()
            return True

def show_troubleshooting():
    with st.expander("🔧 Troubleshooting Guide", expanded=True):
        if STATE["error_info"]:
            st.write(f"**Error Type**: `{STATE['error_info']['type']}`")
            st.write(f"**Step**: {STATE['error_info']['step']}")
            
            st.subheader("Auto Diagnosis")
            diagnosis = llm.chat(
                f"Diagnose this data science error: {STATE['error_info']['message']}"
                "Provide 3 possible solutions in bullet points"
            )
            st.markdown(diagnosis)
            
            st.subheader("Try These Fixes:")
            if "missing values" in STATE["error_info"]['message'].lower():
                st.button("🔄 Auto Handle Missing Values", 
                         on_click=handle_missing_values)
            if "data type" in STATE["error_info"]['message'].lower():
                st.button("🔄 Auto Convert Data Types",
                         on_click=convert_data_types)
            st.button("↩️ Revert to Previous Data Version",
                     on_click=revert_data_version)

# Data Version Control
def handle_missing_values():
    if STATE["data_versions"]:
        agent = get_agent(STATE["data_versions"][-1])
        cleaned_df = agent.chat("Automatically handle missing values using best practices")
        STATE["data_versions"].append(cleaned_df)

def convert_data_types():
    if STATE["data_versions"]:
        agent = get_agent(STATE["data_versions"][-1])
        converted_df = agent.chat("Automatically detect and convert data types")
        STATE["data_versions"].append(converted_df)

def revert_data_version():
    if len(STATE["data_versions"]) > 1:
        STATE["data_versions"].pop()

# Workflow Steps
def step_upload():
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader("Choose CSV/Excel", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            STATE["data_versions"] = [df]
            show_data_characteristics(df)
            
            st.success("Data loaded successfully!")
            st.session_state.current_step = 2
        except Exception as e:
            st.error(f"Invalid file format: {str(e)}")
            st.info("💡 Try: Convert file to UTF-8 CSV format")

def step_clean():
    st.header("2. Data Cleaning & Preparation")
    
    if not STATE["data_versions"]:
        st.warning("Upload data first!")
        return
    
    current_df = STATE["data_versions"][-1]
    agent = get_agent(current_df)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("AI Suggestions")
        suggestions = agent.chat("Suggest 5 data cleaning steps")
        st.markdown(suggestions)
        
        common_tasks = [
            "Handle missing values",
            "Remove duplicates",
            "Encode categorical variables",
            "Normalize numerical features",
            "Fix data type issues"
        ]
        for task in common_tasks:
            if st.button(f"🔧 {task}"):
                cleaned_df = agent.chat(task)
                STATE["data_versions"].append(cleaned_df)
    
    with col2:
        st.subheader("Custom Processing")
        query = st.text_area("Or write your own cleaning query:", 
                            height=100,
                            placeholder="E.g., 'Remove columns with >50% missing values'")
        
        if st.button("▶ Execute"):
            try:
                processed_df = agent.chat(query)
                STATE["data_versions"].append(processed_df)
                st.success("Processing completed!")
                show_data_characteristics(processed_df)
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.markdown("**Suggested Fixes:**")
                st.write(llm.chat(f"Correct this pandas query: {query}"))

def step_model():
    st.header("3. Model Development")
    
    if not STATE["data_versions"]:
        st.warning("Process data first!")
        return
    
    current_df = STATE["data_versions"][-1]
    agent = get_agent(current_df)
    
    # Target selection
    st.subheader("Target Selection")
    binary_cols = [col for col in current_df.columns 
                  if current_df[col].nunique() == 2]
    if binary_cols:
        STATE["target"] = st.selectbox("Select churn column:", binary_cols)
    else:
        st.warning("No binary columns found for classification!")
        st.write("Potential targets suggested by AI:")
        targets = agent.chat("Suggest 3 potential target columns for churn prediction")
        st.markdown(targets)
    
    # Model selection
    st.subheader("Model Configuration")
    models = agent.chat("List 5 best classification models for this data")
    model_choice = st.selectbox("Choose model:", eval(models))
    
    if st.button("🚀 Train Model"):
        try:
            with st.spinner("Training in progress..."):
                model_code = agent.chat(
                    f"Train {model_choice} model for {STATE['target']} prediction. "
                    "Return Python code for training and evaluation."
                )
                exec_env = {"df": current_df, "st": st}
                exec(model_code, exec_env)
                STATE["model"] = exec_env["model"]
                st.success("Model trained successfully!")
                
                # Show evaluation
                st.subheader("Evaluation Metrics")
                if "classification_report" in exec_env:
                    st.write(exec_env["classification_report"])
                if "confusion_matrix" in exec_env:
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay(exec_env["confusion_matrix"]).plot(ax=ax)
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.write("**AI Suggested Fix:**")
            st.write(llm.chat(f"Fix this model training error: {str(e)}"))

# Main App
def main():
    st.set_page_config(
        page_title="AI Churn Analyst",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("Workflow Navigator")
    step = st.sidebar.radio("Go to:", [
        "1. Data Upload",
        "2. Data Cleaning",
        "3. Model Training"
    ])
    
    if st.sidebar.button("🔄 Reset Entire Workflow"):
        reset_state()
    
    if step == "1. Data Upload":
        with WorkflowGuard(1):
            step_upload()
    elif step == "2. Data Cleaning":
        with WorkflowGuard(2):
            step_clean()
    elif step == "3. Model Training":
        with WorkflowGuard(3):
            step_model()

if __name__ == "__main__":
    main()