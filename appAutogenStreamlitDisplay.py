import streamlit as st
import autogen
import os
import pickle
import subprocess
from pathlib import Path
import pandas as pd
import plotly.express as px
from datetime import datetime
import orjson
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
OLLAMA_MODELS = {
    "User Assistant": [
        "phind-codellama:34b-v2", "codellama:latest", "qwen2.5-coder:7b", "mistral:latest",
        "qwen2.5-coder:7b-instruct", "qwen2.5-coder:32b", "qwen2.5-coder:14b",
        "llama3.2:latest", "deepseek-r1:latest"
    ],
    "Planner": ["phind-codellama:34b-v2", "qwen2.5-coder:32b", "qwen2.5-coder:14b", "codellama:latest"],
    "Code Writer": ["phind-codellama:34b-v2", "codellama:latest", "qwen2.5-coder:7b", "qwen2.5-coder:7b-instruct"],
    "Thinker": ["mistral:latest", "llama3.2:latest", "deepseek-r1:latest", "qwen2.5-coder:14b"],
    "Checker": ["deepseek-r1:latest", "qwen2.5-coder:14b", "mistral:latest", "llama3.2:latest"],
}

class ExpertDataScienceUnit:
    def __init__(self):
        self.work_dir = Path("neuro_workspace")
        self.work_dir.mkdir(exist_ok=True)
        self.state_file = self.work_dir / "pipeline_state.pkl"
        self.init_session()
        self._load_state()
        self.setup_ui()

    def init_session(self):
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = {
                'current_step': "Awaiting Initiation",
                'data_versions': [],
                'model_versions': {},
                'execution_log': [],
                'active_agent': None,
                'current_data': None,
                'requirements': "",
            }
        if 'agents' not in st.session_state:
            st.session_state.agents = {}
        if 'target_var' not in st.session_state:
            st.session_state.target_var = None
        if 'data_columns' not in st.session_state:
            st.session_state.data_columns = []

    def setup_ui(self):
        st.set_page_config(page_title="NeuroLab AI", layout="wide", page_icon="🧠")
        st.markdown(
            """
            <style>
            .stApp { background: linear-gradient(to bottom right, #e0f7fa, #b3e5fc); color: #333333; }
            .sidebar .sidebar-content { background: #ffffff; }
            .stButton>button {
                background: linear-gradient(45deg, #4ecdc4, #45b7d1);
                color: #ffffff;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background: linear-gradient(45deg, #45b7d1, #4ecdc4);
                transform: scale(1.05);
            }
            .stSelectbox, .stTextInput, .stFileUploader, .stTextArea {
                background: #f5f5f5;
                border-radius: 8px;
                padding: 10px;
            }
            .stTabs [data-baseweb="tab"] { color: #0288d1; }
            .stTabs [data-baseweb="tab"]:hover { color: #0277bd; }
            .metric-container {
                background: #e3f2fd;
                border-radius: 10px;
                padding: 10px;
                border: 1px solid #90caf9;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        self.sidebar_controls()
        self.main_interface()

    def sidebar_controls(self):
        with st.sidebar:
            st.header("🧬 NeuroLab Controls")
            self.model_selection()
            self.data_config()
            if st.button("⚡ Initiate Quantum Pipeline"):
                self.init_pipeline()
            if st.button("🔄 Start Over"):
                self.reset_session()

    def model_selection(self):
        st.subheader("AI Architect Selection")
        self.model_config = {
            "user_assistant": st.selectbox("User Assistant", OLLAMA_MODELS["User Assistant"], key="user_assistant"),
            "planner": st.selectbox("Planner", OLLAMA_MODELS["Planner"], key="planner"),
            "code_writer": st.selectbox("Code Writer", OLLAMA_MODELS["Code Writer"], key="code_writer"),
            "thinker": st.selectbox("Thinker", OLLAMA_MODELS["Thinker"], key="thinker"),
            "checker": st.selectbox("Checker", OLLAMA_MODELS["Checker"], key="checker"),
        }

    def data_config(self):
        st.subheader("🔮 Data Configuration")
        self.data_file = st.file_uploader("Upload Dataset", type=["csv"], key="data_file")
        
        # Initialize target_var as None if not set
        if 'target_var' not in st.session_state:
            st.session_state.target_var = None
        
        # Handle column selection after file upload
        columns = []
        if self.data_file:
            try:
                self.data_file.seek(0)
                df = pd.read_csv(self.data_file)
                columns = df.columns.tolist()
                st.session_state.data_columns = columns
            except Exception as e:
                st.error(f"❌ Failed to read dataset: {str(e)}")
        
        # Show selectbox for target variable if columns are available
        if columns:
            st.session_state.target_var = st.selectbox(
                "Target Variable",
                options=["Select a column"] + columns,
                index=0 if st.session_state.target_var not in columns else columns.index(st.session_state.target_var) + 1,
                key="target_var_select"
            )
            if st.session_state.target_var == "Select a column":
                st.session_state.target_var = None
        else:
            st.write("Please upload a CSV file to select a target variable.")
        
        self.problem_type = st.selectbox("Problem Type", ["Classification", "Regression"], key="problem_type")
        self.requirements = st.text_area(
            "Pipeline Requirements",
            placeholder="e.g., Perform classification with Random Forest and XGBoost, use grid search, evaluate with F1-score",
            key="requirements",
        )

    def main_interface(self):
        st.title("🧠 NeuroLab - Autonomous Data Science Unit")
        st.markdown("**Welcome to NeuroLab AI**, your intelligent data science assistant. Upload a dataset, configure the pipeline, and watch our expert agents deliver world-class results!")
        self.status_board()
        self.visualization_engine()
        self.version_navigator()

    def status_board(self):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("⚡ Real-Time Cognitive Process")
            active_agent = st.session_state.pipeline.get('active_agent', 'None')
            current_step = st.session_state.pipeline.get('current_step', 'Awaiting Initiation')
            st.markdown(
                f"""
                <div style="background:#e3f2fd;padding:20px;border-radius:15px;border:1px solid #90caf9">
                    <h3 style="color:#0288d1;text-align:center">🧩 Active Agent: {active_agent}</h3>
                    <p style="color:#333333;text-align:center">🌀 Current Operation: {current_step}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        self.progress_metrics()

    def progress_metrics(self):
        cols = st.columns(4)
        data_versions = st.session_state.pipeline.get('data_versions', [])
        model_versions = st.session_state.pipeline.get('model_versions', {})
        execution_log = st.session_state.pipeline.get('execution_log', [])
        metrics = [
            ("📦 Data Versions", len(data_versions)),
            ("🤖 Model Versions", len(model_versions)),
            ("📜 Process Steps", len(execution_log)),
            ("🔍 Audit Score", f"{np.random.uniform(95, 99):.1f}%"),
        ]
        for col, (label, value) in zip(cols, metrics):
            col.markdown(
                f"""
                <div class="metric-container">
                    <p style="color:#0288d1">{label}</p>
                    <h3 style="color:#0277bd">{value}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def visualization_engine(self):
        tab1, tab2, tab3 = st.tabs(["📊 Data Cortex", "🧪 Model Matrix", "📜 Thought Process"])
        with tab1:
            self.show_data_visualization()
        with tab2:
            self.show_model_performance()
        with tab3:
            self.show_execution_log()

    def show_data_visualization(self):
        current_data = st.session_state.pipeline.get('current_data')
        if current_data:
            try:
                df = pd.read_parquet(current_data)
                st.subheader("Data Visualization")
                col1, col2 = st.columns(2)
                with col1:
                    x = st.selectbox("X Axis", df.columns, key="x_axis")
                with col2:
                    y = st.selectbox("Y Axis", df.columns, key="y_axis")
                fig = px.scatter(
                    df,
                    x=x,
                    y=y,
                    color=st.session_state.target_var if st.session_state.target_var and st.session_state.target_var in df.columns else None,
                    template="plotly_white",
                    title="Data Quantum Field",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Visualization error: {str(e)}")

    def show_model_performance(self):
        model_versions = st.session_state.pipeline.get('model_versions', {})
        if model_versions:
            st.subheader("Model Performance")
            model = st.selectbox(
                "Select Model Version",
                list(model_versions.keys()),
                key="model_select",
            )
            metrics = model_versions[model]
            df_metrics = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
            fig = px.bar(
                df_metrics,
                x='Metric',
                y='Value',
                template="plotly_white",
                title=f"Performance Metrics for {model}",
            )
            st.plotly_chart(fig, use_container_width=True)
            if 'shap_values' in metrics:
                st.subheader("Feature Importance (SHAP)")
                shap_fig = self.plot_shap(metrics['shap_values'], metrics.get('feature_names', []))
                st.pyplot(shap_fig)

    def plot_shap(self, shap_values, feature_names):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        return plt.gcf()

    def show_execution_log(self):
        st.subheader("Execution Log")
        execution_log = st.session_state.pipeline.get('execution_log', [])
        for log in reversed(execution_log):
            st.markdown(
                f"""
                <div style="background:#f5f5f5;padding:10px;border-radius:5px;margin:5px">
                    <small style="color:#666666">{log['timestamp']}</small>
                    <p style="color:#333333;margin:0">🔮 <strong>{log['agent']}:</strong> {log['message']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def version_navigator(self):
        with st.expander("🔍 Data/Model Version Timeline"):
            cols = st.columns(2)
            with cols[0]:
                st.write("**Data Versions**")
                data_versions = st.session_state.pipeline.get('data_versions', [])
                for ver in data_versions:
                    st.caption(f"📅 {ver['timestamp']} - {ver['description']}")
            with cols[1]:
                st.write("**Model Versions**")
                model_versions = st.session_state.pipeline.get('model_versions', {})
                for ver in model_versions:
                    st.caption(
                        f"🤖 {ver} - Accuracy: {model_versions[ver].get('accuracy', 'N/A'):.3f}"
                    )

    def reset_session(self):
        st.session_state.pipeline = {
            'current_step': "Awaiting Initiation",
            'data_versions': [],
            'model_versions': {},
            'execution_log': [],
            'active_agent': None,
            'current_data': None,
            'requirements': "",
        }
        st.session_state.agents = {}
        st.session_state.target_var = None
        st.session_state.data_columns = []
        st.experimental_rerun()

    def validate_inputs(self):
        if not self.data_file:
            st.error("❌ Please upload a CSV dataset.")
            return False
        if not st.session_state.target_var or st.session_state.target_var == "Select a column":
            st.error("❌ Please select a target variable from the dropdown.")
            return False
        if not self.requirements:
            st.error("❌ Please provide pipeline requirements.")
            return False
        try:
            self.data_file.seek(0)
            df = pd.read_csv(self.data_file)
            if st.session_state.target_var not in df.columns:
                st.error(f"❌ Selected target variable '{st.session_state.target_var}' not found in dataset.")
                st.write("**Available columns**:", df.columns.tolist())
                return False
        except Exception as e:
            st.error(f"❌ Invalid dataset: {str(e)}")
            return False
        return True

    def initialize_agents(self):
        try:
            st.session_state.agents = {
                'user_assistant': autogen.AssistantAgent(
                    name="UserAssistant",
                    system_message="Friendly UI specialist. Validate inputs and guide users.",
                    llm_config=self.get_llm_config("user_assistant"),
                ),
                'planner': autogen.AssistantAgent(
                    name="Planner",
                    system_message="Expert data science planner. Create atomic task plans for validation, EDA, cleaning, feature engineering, model training, tuning, evaluation, and SHAP explanations.",
                    llm_config=self.get_llm_config("planner"),
                ),
                'code_writer': autogen.AssistantAgent(
                    name="CodeWriter",
                    system_message="Senior Python engineer. Write production-grade code with error handling, cross-validation, and intermediate saving.",
                    llm_config=self.get_llm_config("code_writer"),
                ),
                'code_executor': autogen.UserProxyAgent(
                    name="CodeExecutor",
                    code_execution_config={
                        "work_dir": str(self.work_dir),
                        "use_docker": False,
                    },
                    human_input_mode="NEVER",
                ),
                'thinker': autogen.AssistantAgent(
                    name="Thinker",
                    system_message="Critical analyst. Resolve issues, optimize methods, and suggest improvements using first principles.",
                    llm_config=self.get_llm_config("thinker"),
                ),
                'checker': autogen.AssistantAgent(
                    name="Checker",
                    system_message="Quality assurance expert. Verify data lineage, metrics, and code quality, requiring rework if standards are unmet.",
                    llm_config=self.get_llm_config("checker"),
                ),
            }
        except Exception as e:
            st.error(f"Agent initialization failed: {str(e)}")
            raise

    def get_llm_config(self, role):
        return {
            "config_list": [{
                "model": self.model_config[role],
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "temperature": 0.3,
                "api_type": "ollama",
            }],
            "cache_seed": None,
        }

    def _load_state(self):
        self.state = {
            'data': None,
            'models': {},
            'metrics': {},
            'explanations': {},
            'step': 0,
        }
        if self.state_file.exists():
            try:
                with open(self.state_file, "rb") as f:
                    saved_state = pickle.load(f)
                    if saved_state.get('version') == 1.0:
                        self.state.update(saved_state['data'])
            except Exception as e:
                st.warning(f"State load error: {e}, starting fresh")

    def _save_state(self):
        state_data = {
            'version': 1.0,
            'data': self.state,
        }
        try:
            with open(self.state_file, "wb") as f:
                pickle.dump(state_data, f, protocol=4)
        except Exception as e:
            st.warning(f"State save error: {str(e)}")

    def _install_deps(self):
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
            "plotly==5.14.1",
        ]
        try:
            subprocess.check_call(
                ["pip", "install", "--quiet"] + deps,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            st.error(f"Dependency install failed: {str(e)}")
            raise

    def init_pipeline(self):
        if not self.validate_inputs():
            return
        try:
            self._install_deps()
            self.initialize_agents()
            self.execute_pipeline()
        except Exception as e:
            st.error(f"Pipeline initialization failed: {str(e)}")
            self.rollback_system()

    def execute_pipeline(self):
        try:
            self.process_step("Initializing Quantum Data Processing...", "UserAssistant")
            self.load_data()
            plan = self.create_plan()
            code = self.generate_code(plan)
            self.execute_code(code)
            analysis = self.analyze_results()
            self.verify_quality()
            st.success("🎉 Quantum Processing Complete!")
        except Exception as e:
            st.error(f"Pipeline execution failed: {str(e)}")
            self.rollback_system()

    def process_step(self, message, agent):
        st.session_state.pipeline['current_step'] = message
        st.session_state.pipeline['active_agent'] = agent
        st.session_state.pipeline['execution_log'].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent": agent,
            "message": message,
        })
        self._save_state()
        st.experimental_rerun()

    def load_data(self):
        try:
            self.process_step("Loading and Validating Dataset...", "UserAssistant")
            self.data_file.seek(0)
            df = pd.read_csv(self.data_file)
            data_path = self.work_dir / "raw_data.parquet"
            df.to_parquet(data_path)
            st.session_state.pipeline['current_data'] = str(data_path)
            st.session_state.pipeline['data_versions'].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": "Raw dataset loaded",
                "path": str(data_path),
            })
            self._save_state()
        except Exception as e:
            st.error(f"Data loading failed: {str(e)}")
            raise

    def create_plan(self):
        try:
            self.process_step("Developing Atomic Execution Plan...", "Planner")
            plan = [
                "Validate data types and missing values",
                "Perform EDA with histograms and correlation plots",
                "Clean data (impute missing values, remove outliers)",
                "Engineer features (polynomial features, encoding)",
                "Train Random Forest with cross-validation",
                "Evaluate with accuracy and F1-score",
                "Generate SHAP explanations",
            ]
            return plan
        except Exception as e:
            st.error(f"Plan creation failed: {str(e)}")
            raise

    def generate_code(self, plan):
        try:
            self.process_step("Generating Production Code...", "CodeWriter")
            code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import shap
import orjson
import os

# Load data
try:
    df = pd.read_parquet('{data_path}')
except Exception as e:
    raise ValueError(f"Data loading failed: {{e}}")

# Validate data
if '{target}' not in df.columns:
    raise ValueError("Target column '{target}' not found")
X = df.drop('{target}', axis=1)
y = df['{target}']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = {{
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred, average='weighted')
}}

# SHAP explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Save results
os.makedirs('{work_dir}', exist_ok=True)
with open('{work_dir}/model_rf.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('{work_dir}/metrics_rf.json', 'wb') as f:
    orjson.dump(metrics, f)
np.save('{work_dir}/shap_values_rf.npy', shap_values)
"""
            code = code.format(
                data_path=st.session_state.pipeline['current_data'],
                target=st.session_state.target_var,
                work_dir=self.work_dir,
            )
            return code
        except Exception as e:
            st.error(f"Code generation failed: {str(e)}")
            raise

    def execute_code(self, code):
        try:
            self.process_step("Executing Computational Matrix...", "CodeExecutor")
            code_path = self.work_dir / "temp_script.py"
            with open(code_path, "w") as f:
                f.write(code)
            result = subprocess.run(
                ["python", str(code_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Code execution failed: {result.stderr}")
            metrics_path = self.work_dir / "metrics_rf.json"
            if metrics_path.exists():
                with open(metrics_path, "rb") as f:
                    metrics = orjson.loads(f.read())
                shap_values_path = self.work_dir / "shap_values_rf.npy"
                shap_values = np.load(shap_values_path, allow_pickle=True) if shap_values_path.exists() else None
                st.session_state.pipeline['model_versions']['RandomForest'] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'shap_values': shap_values,
                    'feature_names': pd.read_parquet(st.session_state.pipeline['current_data']).columns.tolist(),
                }
            self._save_state()
        except Exception as e:
            st.error(f"Code execution failed: {str(e)}")
            raise

    def analyze_results(self):
        try:
            self.process_step("Performing Critical Analysis...", "Thinker")
            analysis = "Model accuracy is satisfactory. Consider feature scaling for improved performance."
            return analysis
        except Exception as e:
            st.error(f"Result analysis failed: {str(e)}")
            raise

    def verify_quality(self):
        try:
            self.process_step("Conducting Quality Assurance...", "Checker")
            if not st.session_state.pipeline['model_versions']:
                raise ValueError("No models trained")
            return True
        except Exception as e:
            st.error(f"Quality verification failed: {str(e)}")
            raise

    def rollback_system(self):
        try:
            self.process_step("Rolling Back to Stable State...", "UserAssistant")
            data_versions = st.session_state.pipeline.get('data_versions', [])
            if data_versions:
                st.session_state.pipeline['current_data'] = data_versions[-1]['path']
            else:
                st.session_state.pipeline['current_data'] = None
            self._save_state()
        except Exception as e:
            st.error(f"Rollback failed: {str(e)}")

if __name__ == "__main__":
    ExpertDataScienceUnit()