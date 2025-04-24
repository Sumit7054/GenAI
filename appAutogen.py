from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
import pickle
import subprocess
from pathlib import Path
import requests
import sklearn
import pandas as pd
# Ollama model configuration
CONFIG_LIST = [
    {
        "model": "phind-codellama:34b-v2",  # Best for code generation
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    {
        "model": "deepseek-r1",  # Best for chat/completion
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
]

class DataSciencePipeline:
    def __init__(self):
        self.work_dir = Path("ds_workspace")
        self.work_dir.mkdir(exist_ok=True)
        self.state_file = self.work_dir / "pipeline_state.pkl"
        self.agents = {}
        self._init_agents()
        self._load_state()

    def _init_agents(self):
        """Initialize all agents with role-specific configurations"""
        # User Proxy Agent
        self.agents['user_proxy'] = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="TERMINATE",
            code_execution_config={
                "work_dir": str(self.work_dir),
                "use_docker": False,
            },
        )

        # Planner Agent (using CodeLlama)
        self.agents['planner'] = AssistantAgent(
            name="Planner",
            system_message="""Expert data science planner. Use first principles to create detailed task plans including:
            1. Data validation 2. EDA 3. Cleaning 4. Feature engineering
            5. Model training (Logistic, RF, XGBoost, SVM, MLP)
            6. Hyperparameter tuning (Grid/Random/Bayesian)
            7. Evaluation metrics (AUC, F1, Precision/Recall)
            8. SHAP/LIME explanations
            Output must be atomic executable steps.""",
            llm_config={"config_list": [CONFIG_LIST[0]], "cache_seed": None},
        )

        # Coder Agent (using CodeLlama)
        self.agents['coder'] = AssistantAgent(
            name="Coder",
            system_message="""Senior Python data engineer. Write production-grade code with:
            - Full error handling - Version compatibility - Dependency management
            - Efficient memory usage - Intermediate saving
            - Cross-validation - Feature importance
            Always validate data shape before processing.""",
            llm_config={"config_list": [CONFIG_LIST[0]], "cache_seed": None},
        )

        # Analyst Agent (using DeepSeek)
        self.agents['analyst'] = AssistantAgent(
            name="Analyst",
            system_message="""Lead data scientist. Perform rigorous analysis:
            - Statistical testing - Model diagnostics
            - Bias detection - Feature significance
            - Hyperparameter impact - Error analysis
            Use SHAP/LIME for explanations.""",
            llm_config={"config_list": [CONFIG_LIST[1]], "cache_seed": None},
        )

        # Quality Agent (using DeepSeek)
        self.agents['quality'] = AssistantAgent(
            name="Quality_Check",
            system_message="""Senior ML engineer. Verify:
            - Data lineage - Model metrics - Code quality
            - Error propagation - Concept drift
            - Training/validation splits
            Require rework if standards not met.""",
            llm_config={"config_list": [CONFIG_LIST[1]], "cache_seed": None},
        )

    def _load_state(self):
        """Load pipeline state with version checking"""
        self.state = {
            'data': None,
            'models': {},
            'metrics': {},
            'explanations': {},
            'step': 0
        }
        if self.state_file.exists():
            try:
                with open(self.state_file, "rb") as f:
                    saved_state = pickle.load(f)
                    if saved_state.get('version') == 1.0:
                        self.state.update(saved_state['data'])
            except Exception as e:
                print(f"State load error: {e}, starting fresh")

    def _save_state(self):
        """Save pipeline state with versioning"""
        state_data = {
            'version': 1.0,
            'data': self.state
        }
        with open(self.state_file, "wb") as f:
            pickle.dump(state_data, f, protocol=4)

    def _install_deps(self):
        """Ensure required dependencies are installed"""
        deps = [
            "pandas==1.5.3",
            "numpy==1.23.5",
            "scikit-learn==1.2.2",
            "xgboost==1.7.6",
            "lightgbm==3.3.5",
            "shap==0.42.1",
            "imbalanced-learn==0.10.1",
            "seaborn==0.12.2",
            "matplotlib==3.7.1",
        ]
        try:
            subprocess.check_call(
                ["pip", "install", "--quiet"] + deps,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"Dependency install failed: {e}")
            raise

    def _check_ollama_server(self):
        """Verify Ollama server is running"""
        try:
            response = requests.get("http://localhost:11434")
            if response.status_code != 200:
                raise Exception("Ollama server not responding")
        except Exception as e:
            print(f"Ollama server error: {e}")
            print("Ensure 'ollama serve' is running and port 11434 is accessible")
            raise

    def run(self):
        """Execute full pipeline"""
        self._check_ollama_server()
        self._install_deps()
        
        group_chat = GroupChat(
            agents=list(self.agents.values()),
            messages=[],
            max_round=40,
            speaker_selection_method="round_robin",
        )
        
        # Initialize GroupChatManager with Ollama llm_config
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config={"config_list": CONFIG_LIST, "cache_seed": None}
        )
        
        try:
            self.agents['user_proxy'].initiate_chat(
                manager,
                message="""Begin data science pipeline. First collect:
                1. Dataset path (CSV)
                2. Target variable
                3. Problem type (classification/regression)
                4. Validation strategy"""
            )
            self._save_state()
        except Exception as e:
            print(f"Pipeline failed: {e}")
            print("Saved state available for debugging at:", self.state_file)

if __name__ == "__main__":
    pipeline = DataSciencePipeline()
    pipeline.run()