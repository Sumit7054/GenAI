import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import traceback
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, 
                                  MinMaxScaler, RobustScaler,
                                  FunctionTransformer, PolynomialFeatures,
                                  OrdinalEncoder)
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                    RandomizedSearchCV, GridSearchCV,
                                    cross_val_score)
from sklearn.ensemble import (RandomForestClassifier, 
                             GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            roc_curve, auc, precision_recall_curve,
                            classification_report, make_scorer)
from sklearn.feature_selection import (RFECV, SelectFromModel,
                                      mutual_info_classif, SelectKBest,
                                      f_classif)
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import lime
import lime.lime_tabular
import shap
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM
import warnings
warnings.filterwarnings('ignore')

# Initialize LLM (fallback to simple analysis if not available)
try:
    llm = LocalLLM(api_base="http://localhost:11434/v1", model="mistral")
except:
    llm = None
    st.sidebar.warning("LLM not available - some AI features disabled")

# Session state management
class SessionState:
    def __init__(self):
        self.df = None
        self.target = None
        self.features = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.current_step = 0
        self.feature_names = []
        
def get_session():
    if 'session' not in st.session_state:
        st.session_state.session = SessionState()
    return st.session_state.session

state = get_session()

# UI Configuration
st.set_page_config(
    page_title="Employee Attrition AI Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .st-bd {
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Data Management
def handle_data_loading():
    st.header("1. Data Management")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        source = st.radio("Data Source:", 
                        ["Use Sample Dataset", "Upload Custom Data"],
                        key="data_source")
        
        if source == "Upload Custom Data":
            uploaded_file = st.file_uploader("Choose CSV", 
                                           type=["csv"],
                                           key="file_uploader")
            if uploaded_file:
                try:
                    state.df = pd.read_csv(uploaded_file)
                    st.success("Data loaded successfully!")
                    state.target = None
                    state.features = None
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        else:
            try:
                state.df = pd.read_csv("https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv")
                st.success("Default dataset loaded!")
                state.target = "Attrition"
                state.features = [col for col in state.df.columns if col != "Attrition"]
            except Exception as e:
                st.error(f"Error loading default data: {str(e)}")
    
    with col2:
        if state.df is not None:
            st.subheader("Data Summary")
            st.write(f"Rows: {state.df.shape[0]}")
            st.write(f"Columns: {state.df.shape[1]}")
            
            if st.button("Show Sample Data", key="sample_data_btn"):
                st.dataframe(state.df.head())
            
            if st.button("Clean Data Automatically", key="auto_clean_btn"):
                try:
                    # Handle missing values
                    num_cols = state.df.select_dtypes(include=np.number).columns
                    cat_cols = state.df.select_dtypes(exclude=np.number).columns
                    
                    state.df[num_cols] = state.df[num_cols].fillna(state.df[num_cols].median())
                    state.df[cat_cols] = state.df[cat_cols].fillna(state.df[cat_cols].mode().iloc[0])
                    
                    # Remove duplicates
                    state.df = state.df.drop_duplicates()
                    st.success("Data cleaned automatically!")
                except Exception as e:
                    st.error(f"Cleaning failed: {str(e)}")

# Feature Management
def handle_feature_engineering():
    st.header("2. Feature Configuration")
    
    if state.df is None:
        st.warning("Please load data first!")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Target Selection")
        state.target = st.selectbox("Select Target Variable", 
                                  state.df.columns,
                                  index=0 if not state.target else list(state.df.columns).index(state.target),
                                  key="target_select")
        
        available_features = [col for col in state.df.columns if col != state.target]
        state.features = st.multiselect("Select Features", 
                                      available_features,
                                      default=available_features if not state.features else state.features,
                                      key="feature_select")
        
        st.subheader("Data Types")
        num_cols = st.multiselect("Numeric Features",
                                state.features,
                                default=[col for col in state.features if pd.api.types.is_numeric_dtype(state.df[col])],
                                key="num_cols")
        
        cat_cols = st.multiselect("Categorical Features",
                                state.features,
                                default=[col for col in state.features if col not in num_cols],
                                key="cat_cols")
    
    with col2:
        st.subheader("Preprocessing Pipeline")
        
        # Numeric preprocessing
        num_strategy = st.selectbox("Numeric Handling:",
                                  ["Standard Scaler", "MinMax Scaler", "Robust Scaler", "None"],
                                  key="num_strategy")
        
        num_impute = st.selectbox("Numeric Imputation:",
                                ["Median", "Mean", "Constant"],
                                key="num_impute")
        
        # Categorical preprocessing
        cat_strategy = st.selectbox("Categorical Encoding:",
                                  ["One-Hot", "Ordinal", "Frequency"],
                                  key="cat_strategy")
        
        cat_impute = st.selectbox("Categorical Imputation:",
                                ["Mode", "Constant"],
                                key="cat_impute")
        
        if st.button("Build Preprocessing Pipeline", key="build_pipe_btn"):
            try:
                numeric_transformer = []
                if num_impute == "Median":
                    numeric_transformer.append(('imputer', SimpleImputer(strategy='median')))
                elif num_impute == "Mean":
                    numeric_transformer.append(('imputer', SimpleImputer(strategy='mean')))
                else:
                    numeric_transformer.append(('imputer', SimpleImputer(strategy='constant', fill_value=0)))
                
                if num_strategy == "Standard Scaler":
                    numeric_transformer.append(('scaler', StandardScaler()))
                elif num_strategy == "MinMax Scaler":
                    numeric_transformer.append(('scaler', MinMaxScaler()))
                elif num_strategy == "Robust Scaler":
                    numeric_transformer.append(('scaler', RobustScaler()))
                
                categorical_transformer = []
                if cat_impute == "Mode":
                    categorical_transformer.append(('imputer', SimpleImputer(strategy='most_frequent')))
                else:
                    categorical_transformer.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
                
                if cat_strategy == "One-Hot":
                    categorical_transformer.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
                elif cat_strategy == "Ordinal":
                    categorical_transformer.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
                else:
                    categorical_transformer.append(('encoder', FunctionTransformer(lambda x: x.astype('category').cat.codes)))
                
                preprocessor = ColumnTransformer([
                    ('num', Pipeline(numeric_transformer), num_cols),
                    ('cat', Pipeline(categorical_transformer), cat_cols)
                ], remainder='passthrough')
                
                # Fit and transform
                X = state.df[state.features]
                y = state.df[state.target]
                
                X_processed = preprocessor.fit_transform(X)
                
                # Get feature names
                num_features = preprocessor.named_transformers_['num'].get_feature_names_out(num_cols)
                cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
                state.feature_names = list(num_features) + list(cat_features)
                
                state.X_processed = pd.DataFrame(X_processed, columns=state.feature_names)
                state.y = y
                state.preprocessor = preprocessor
                
                st.success("Preprocessing completed!")
                st.dataframe(state.X_processed.head())
                
            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")

# Model Training
def handle_model_training():
    st.header("3. Model Development")
    
    if state.X_processed is None:
        st.warning("Preprocess data first!")
        return
    
    model_config = {
        "Random Forest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 100,
                "random_state": 42
            }
        },
        "XGBoost": {
            "class": GradientBoostingClassifier,
            "params": {
                "learning_rate": 0.1,
                "n_estimators": 100,
                "random_state": 42
            }
        },
        "Logistic Regression": {
            "class": LogisticRegression,
            "params": {
                "max_iter": 1000,
                "random_state": 42
            }
        },
        "Neural Network": {
            "class": MLPClassifier,
            "params": {
                "hidden_layer_sizes": (100,),
                "random_state": 42
            }
        }
    }
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Model Selection")
        selected_models = st.multiselect("Choose Models", 
                                       list(model_config.keys()),
                                       key="model_select")
        
        test_size = st.slider("Test Size (%)", 10, 40, 20, key="test_size")
        random_state = st.number_input("Random State", 42, key="random_state")
        
        if st.button("Train Models", key="train_models_btn"):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    state.X_processed, state.y, 
                    test_size=test_size/100, 
                    random_state=random_state,
                    stratify=state.y
                )
                
                state.X_train = X_train
                state.X_test = X_test
                state.y_train = y_train
                state.y_test = y_test
                
                progress_bar = st.progress(0)
                results = []
                
                for idx, model_name in enumerate(selected_models):
                    progress_bar.progress((idx + 1) / len(selected_models))
                    config = model_config[model_name]
                    model = config["class"](**config["params"])
                    model.fit(X_train, y_train)
                    
                    # Calculate metrics
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                    
                    metrics = {
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred, average='weighted'),
                        "Recall": recall_score(y_test, y_pred, average='weighted'),
                        "F1": f1_score(y_test, y_pred, average='weighted')
                    }
                    
                    if y_proba is not None:
                        metrics["ROC AUC"] = roc_auc_score(y_test, y_proba)
                    
                    state.models[model_name] = {
                        "instance": model,
                        "metrics": metrics,
                        "predictions": y_pred,
                        "probabilities": y_proba
                    }
                    results.append(metrics)
                
                st.success("Training completed!")
                progress_bar.empty()
                
                # Display metrics
                st.subheader("Model Performance")
                metrics_df = pd.DataFrame({
                    model_name: state.models[model_name]["metrics"]
                    for model_name in selected_models
                }).T
                st.dataframe(metrics_df.style.background_gradient(cmap='Blues'))
                
            except Exception as e:
                st.error(f"Training error: {str(e)}")
    
    with col2:
        if state.models:
            st.subheader("Model Analysis")
            selected_model = st.selectbox("Choose Model", 
                                        list(state.models.keys()),
                                        key="model_analysis_select")
            
            model = state.models[selected_model]["instance"]
            y_pred = state.models[selected_model]["predictions"]
            y_proba = state.models[selected_model]["probabilities"]
            
            # Confusion Matrix
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(state.y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
            # ROC Curve
            if y_proba is not None:
                st.markdown("### ROC Curve")
                fpr, tpr, _ = roc_curve(state.y_test, y_proba, pos_label="Yes")
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend()
                st.pyplot(fig)

# Explainability
def handle_explainability():
    st.header("4. Model Insights")
    
    if not state.models:
        st.warning("Train models first!")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configuration")
        selected_model = st.selectbox("Select Model", 
                                    list(state.models.keys()),
                                    key="explain_model_select")
        method = st.selectbox("Explanation Method",
                            ["Feature Importance", "SHAP", "LIME"],
                            key="explain_method_select")
        sample_idx = st.number_input("Sample Index", 
                                   0, len(state.X_test)-1, 0,
                                   key="sample_idx_input")
    
    with col2:
        st.subheader("Explanation")
        
        try:
            model = state.models[selected_model]["instance"]
            X_test = state.X_test
            
            if method == "Feature Importance":
                if hasattr(model, "feature_importances_"):
                    importances = pd.Series(model.feature_importances_, 
                                          index=state.feature_names)
                    importances = importances.sort_values(ascending=False)
                    
                    fig, ax = plt.subplots()
                    importances.head(20).plot.bar(ax=ax)
                    ax.set_title("Feature Importances")
                    st.pyplot(fig)
                else:
                    st.warning("Feature importance not available for this model")
            
            elif method == "SHAP":
                explainer = shap.Explainer(model)
                shap_values = explainer(state.X_processed)
                
                # Summary plot
                st.markdown("### Global Feature Impact")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, state.X_processed)
                st.pyplot(fig)
                
                # Local explanation
                st.markdown("### Local Explanation")
                sample = state.X_test.iloc[sample_idx:sample_idx+1]
                shap_value = explainer(sample)
                
                fig, ax = plt.subplots()
                shap.plots.waterfall(shap_value[0], max_display=15)
                st.pyplot(fig)
            
            elif method == "LIME":
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    state.X_train.values,
                    feature_names=state.feature_names,
                    class_names=state.y.unique(),
                    mode='classification'
                )
                
                exp = explainer.explain_instance(
                    state.X_test.iloc[sample_idx].values,
                    model.predict_proba,
                    num_features=15
                )
                
                st.markdown("### Local Explanation")
                st.code(exp.as_list())
                
        except Exception as e:
            st.error(f"Explanation error: {str(e)}")

# AI Assistant
def handle_ai_assistant():
    st.header("5. AI Assistant")
    
    if state.df is None:
        st.warning("Load data first!")
        return
    
    query = st.text_area("Ask anything about your data or models:", 
                       height=100,
                       key="ai_query_input")
    
    if st.button("Get Analysis", key="ai_analyze_btn"):
        try:
            if llm is None:
                st.warning("LLM service not available")
                return
                
            # Create a safe dataframe wrapper
            safe_df = state.df.copy()
            safe_df.columns = [col.replace("__", "") for col in safe_df.columns]
            
            # Initialize SmartDataframe with error handling
            sdf = SmartDataframe(
                safe_df,
                config={
                    "llm": llm,
                    "enable_cache": False,
                    "custom_whitelisted_dependencies": ["sklearn"]
                }
            )
            
            # Process query with validation
            if not query.strip():
                st.warning("Please enter a valid question")
                return
                
            sanitized_query = query.replace("__", "").strip()
            
            # Get response with enhanced error handling
            try:
                response = sdf.chat(
                    sanitized_query,
                    output_type="string"
                )
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                return
            
            # Display results
            st.subheader("AI Analysis")
            st.markdown(f"```\n{response}\n```")
            
            # Handle visualizations safely
            try:
                if hasattr(sdf.last_result, 'plot'):
                    fig = sdf.last_result.plot()
                    st.pyplot(fig)
                elif hasattr(sdf.last_result, 'figure'):
                    st.pyplot(sdf.last_result.figure)
            except Exception as e:
                st.warning(f"Visualization error: {str(e)}")
                
        except Exception as e:
            st.error(f"Critical error in AI analysis: {str(e)}")
            st.write("Common solutions:")
            st.markdown("""
                1. Check LLM server is running (Ollama/Mistral)
                2. Avoid special characters in questions
                3. Simplify complex queries
                4. Check dataset contains valid data
            """)

# Main Application Flow
def main():
    st.sidebar.title("Navigation")
    app_pages = {
        "Data Management": handle_data_loading,
        "Feature Engineering": handle_feature_engineering,
        "Model Training": handle_model_training,
        "Model Insights": handle_explainability,
        "AI Assistant": handle_ai_assistant
    }
    
    selected_page = st.sidebar.radio("Go to", list(app_pages.keys()))
    app_pages[selected_page]()
    
    st.sidebar.title("Session Info")
    if state.df is not None:
        st.sidebar.markdown(f"""
            **Data Info**
            - Rows: {state.df.shape[0]}
            - Columns: {state.df.shape[1]}
            - Target: {state.target or 'Not set'}
        """)
    
    if state.models:
        st.sidebar.markdown("**Trained Models**")
        for model_name in state.models:
            st.sidebar.markdown(f"- {model_name}")

if __name__ == "__main__":
    main()