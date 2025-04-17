# app.py
import streamlit as st
import pandas as pd
import numpy as np
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager
from pandasai import SmartDataframe
import ollama
import openai
from pandasai.llm.local_llm import LocalLLM

# Initialize LLM (fallback to simple analysis if not available)
try:
    llm = LocalLLM(api_base="http://localhost:11434/v1", model="mistral")
except:
    llm = None
    st.sidebar.warning("LLM not available - some AI features disabled")

# Initialize AutoGen agents
llm_config = {
    "config_list": [
        {
            "model": "llama3.2",
            "base_url": "http://localhost:11434/v1",
            "api_type": "openai",
            "api_key": "NULL",
        }, {"model": "mistral",
            "base_url": "http://localhost:11434/v1",
            "api_type": "openai",
            "api_key": "NULL"}],
    "timeout": 300,
    "cache_seed": 42
}

class AgentSystem:
    def __init__(self):
        # Define agents
        self.data_analyst = AssistantAgent(
            name="Data_Analyst",
            system_message="You are a data expert. Handle data loading, cleaning, preprocessing. Generate Python code.",
            llm_config=llm_config
        )
        
        self.ml_engineer = AssistantAgent(
            name="ML_Engineer",
            system_message="You are an ML expert. Handle feature engineering, model selection, hyperparameter tuning.",
            llm_config=llm_config
        )
        
        self.viz_specialist = AssistantAgent(
            name="Viz_Specialist",
            system_message="You create visualizations. Generate matplotlib/seaborn code.",
            llm_config=llm_config
        )
        
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            code_execution_config={"work_dir": "coding"},
            default_auto_reply="Continue",
            max_consecutive_auto_reply=5
        )
        
        # Configure group chat
        self.group_chat = autogen.GroupChat(
            agents=[self.user_proxy, self.data_analyst, self.ml_engineer, self.viz_specialist],
            messages=[],
            max_round=20
        )
        
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=llm_config
        )

    def process_request(self, task):
        self.user_proxy.initiate_chat(
            self.manager,
            message=task,
            clear_history=False
        )
        return self.group_chat.last_message()["content"]

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI-Powered ML Studio",
        page_icon="🤖",
        layout="wide"
    )
    
    agent_system = AgentSystem()
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    
    # Sidebar controls
    with st.sidebar:
        st.header("Data Configuration")
        data_source = st.radio(
            "Select Data Source",
            ["Sample Data", "Upload CSV"]
        )
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Choose CSV")
            if uploaded_file:
                st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_csv(
                "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
            )
        
        st.header("Task Selection")
        task = st.selectbox(
            "Choose Task",
            ["Auto Clean Data", 
             "Feature Engineering",
             "Train Models",
             "Generate Visualizations",
             "Custom Analysis"]
        )
    
    # Main interface
    st.header("AI-Powered Machine Learning Studio")
    
    # Task handling
    if task == "Auto Clean Data":
        if st.button("Run Data Cleaning"):
            response = agent_system.process_request(
                f"Clean this dataset: {st.session_state.df.head().to_csv()}"
            )
            if "```python" in response:
                code = response.split("```python")[1].split("```")[0]
                try:
                    exec(code)
                    st.session_state.df = locals().get('cleaned_df')
                    st.success("Data cleaned successfully!")
                    st.dataframe(st.session_state.df.head())
                except Exception as e:
                    st.error(f"Execution error: {str(e)}")
    
    elif task == "Feature Engineering":
        target = st.selectbox("Select Target Variable", st.session_state.df.columns)
        if st.button("Generate Features"):
            response = agent_system.process_request(
                f"Suggest feature engineering for predicting {target} using\n{st.session_state.df.head().to_csv()}"
            )
            st.code(response)
    
    elif task == "Train Models":
        target = st.selectbox("Select Target Variable", st.session_state.df.columns)
        model_type = st.selectbox("Select Model Type", 
                                ["Random Forest", "XGBoost", "Logistic Regression"])
        
        if st.button("Start Training"):
            with st.spinner("Training in progress..."):
                response = agent_system.process_request(
                    f"Train a {model_type} model to predict {target} using\n{st.session_state.df.head().to_csv()}"
                )
                
                if "```python" in response:
                    code = response.split("```python")[1].split("```")[0]
                    try:
                        exec(code)
                        model = locals().get('trained_model')
                        st.session_state.models[model_type] = model
                        st.success(f"{model_type} trained successfully!")
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
    
    elif task == "Generate Visualizations":
        viz_query = st.text_input("Enter visualization request")
        if st.button("Generate Plot"):
            sdf = SmartDataframe(st.session_state.df, config={"llm": llm})
            try:
                fig = sdf.chat(viz_query, output_type="plot")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
    
    elif task == "Custom Analysis":
        custom_query = st.text_area("Enter your analysis request")
        if st.button("Execute"):
            sdf = SmartDataframe(st.session_state.df, config={"llm": llm})
            try:
                response = sdf.chat(custom_query)
                st.write(response)
                if hasattr(sdf.last_result, 'plot'):
                    st.pyplot(sdf.last_result.plot())
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()