import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
import lime
import lime.lime_tabular
import shap
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import io
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize LLM (using free proxy for OpenAI)
llm = OpenAI(api_token="free", model="gpt-3.5-turbo")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stSelectbox, .stMultiselect, .stSlider, .stTextInput {
        margin-bottom: 20px;
    }
    .css-1aumxhk {
        background-color: #4CAF50;
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title("👥 Employee Attrition Prediction Dashboard")
st.markdown("""
    This application helps predict employee attrition using machine learning. 
    Upload your dataset or use the default IBM HR Analytics dataset.
    """)

# Sidebar for user inputs
with st.sidebar:
    st.header("📊 Data Configuration")
    data_source = st.radio("Select data source:", 
                         ["Use default dataset", "Upload your own CSV"])
    
    if data_source == "Upload your own CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    else:
        st.info("Using default IBM HR Analytics dataset")
    
    st.header("⚙️ Model Settings")
    test_size = st.slider("Test set size (%)", 10, 40, 20)
    random_state = st.number_input("Random state", value=42)
    n_splits = st.number_input("Number of CV folds", min_value=2, max_value=10, value=5)
    
    st.header("🔍 Feature Selection")
    feature_selection_method = st.selectbox(
        "Feature selection method:",
        ["None", "SelectKBest", "RFE", "Correlation Threshold"]
    )
    
    if feature_selection_method in ["SelectKBest", "RFE"]:
        n_features = st.number_input("Number of features to select", min_value=1, value=5)

# Load data
@st.cache_data
def load_default_data():
    try:
        # Try to load from URL
        url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
        df = pd.read_csv(url)
        return df
    except:
        # Fallback to local file if URL fails
        import base64
        import io
        default_data = """Age,Attrition,BusinessTravel,DailyRate,Department,DistanceFromHome,Education,EducationField,EmployeeCount,EmployeeNumber,EnvironmentSatisfaction,Gender,HourlyRate,JobInvolvement,JobLevel,JobRole,JobSatisfaction,MaritalStatus,MonthlyIncome,MonthlyRate,NumCompaniesWorked,Over18,OverTime,PercentSalaryHike,PerformanceRating,RelationshipSatisfaction,StandardHours,StockOptionLevel,TotalWorkingYears,TrainingTimesLastYear,WorkLifeBalance,YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,YearsWithCurrManager
41,Yes,Travel_Rarely,1102,Sales,1,2,Life Sciences,1,1,2,Female,94,3,2,Sales Executive,4,Single,5993,19479,8,Y,Yes,11,3,1,80,1,8,0,1,6,4,0,5
49,No,Travel_Frequently,279,Research & Development,8,1,Life Sciences,1,2,3,Male,61,2,2,Research Scientist,2,Married,5130,24907,1,Y,No,23,4,4,80,0,10,3,3,10,7,1,7
37,Yes,Travel_Rarely,1373,Research & Development,2,2,Other,1,3,4,Male,92,2,1,Laboratory Technician,3,Single,2090,2396,6,Y,Yes,15,3,2,80,0,7,3,3,0,0,0,0"""
        return pd.read_csv(io.StringIO(default_data))

def load_data():
    if data_source == "Upload your own CSV" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        return df
    else:
        df = load_default_data()
        st.session_state['df'] = df
        return df

try:
    df = load_data()
    
    # Display raw data
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df.head())
        
    # Get user input for target and features
    with st.sidebar:
        if 'df' in st.session_state:
            all_columns = st.session_state['df'].columns.tolist()
            target_col = st.selectbox("Select target variable (Attrition):", all_columns, 
                                    index=all_columns.index('Attrition') if 'Attrition' in all_columns else 0)
            
            default_features = [col for col in all_columns if col != target_col]
            selected_features = st.multiselect("Select features:", all_columns, 
                                             default=default_features)
            
            # Numeric and categorical columns
            numeric_cols = st.session_state['df'].select_dtypes(include=np.number).columns.tolist()
            categorical_cols = st.session_state['df'].select_dtypes(exclude=np.number).columns.tolist()
            
            # Remove target from numeric/categorical if it's there
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            if target_col in categorical_cols:
                categorical_cols.remove(target_col)
            
            # Let user adjust
            st.markdown("**Adjust column types:**")
            numeric_cols = st.multiselect("Numeric columns:", all_columns, 
                                         default=numeric_cols)
            categorical_cols = st.multiselect("Categorical columns:", all_columns, 
                                            default=categorical_cols)
            
            st.session_state['numeric_cols'] = numeric_cols
            st.session_state['categorical_cols'] = categorical_cols
            st.session_state['target_col'] = target_col
            st.session_state['selected_features'] = selected_features
    
    # Data Cleaning with PandasAI
    st.header("🧹 Data Cleaning")
    if st.button("Clean Data with AI"):
        with st.spinner("Cleaning data with AI..."):
            try:
                sdf = SmartDataframe(df, config={"llm": llm})
                cleaned_df = sdf.clean_data()
                st.session_state['df'] = cleaned_df
                st.success("Data cleaned successfully!")
                st.dataframe(cleaned_df.head())
            except Exception as e:
                st.error(f"Error cleaning data: {str(e)}")
                st.session_state['df'] = df
    
    # EDA with PandasAI
    st.header("📊 Exploratory Data Analysis")
    eda_options = st.multiselect("Select EDA visualizations:", 
                                ["Correlation Heatmap", "Target Distribution", 
                                 "Feature Distributions", "Pair Plots", 
                                 "Categorical Analysis", "Missing Values"])
    
    if st.button("Generate EDA with AI"):
        with st.spinner("Generating EDA visualizations..."):
            try:
                sdf = SmartDataframe(st.session_state['df'], config={"llm": llm})
                
                if "Correlation Heatmap" in eda_options:
                    st.subheader("Correlation Heatmap")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    numeric_df = st.session_state['df'][st.session_state['numeric_cols']]
                    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", ax=ax)
                    st.pyplot(fig)
                
                if "Target Distribution" in eda_options:
                    st.subheader("Target Variable Distribution")
                    fig, ax = plt.subplots()
                    st.session_state['df'][st.session_state['target_col']].value_counts().plot(kind='bar', ax=ax)
                    st.pyplot(fig)
                
                if "Feature Distributions" in eda_options:
                    st.subheader("Numeric Feature Distributions")
                    num_cols = st.session_state['numeric_cols']
                    n_cols = 3
                    n_rows = (len(num_cols) // n_cols + 1
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4))
                    axes = axes.flatten()
                    
                    for i, col in enumerate(num_cols):
                        sns.histplot(st.session_state['df'][col], kde=True, ax=axes[i])
                        axes[i].set_title(col)
                    
                    for j in range(i+1, len(axes)):
                        axes[j].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                if "Pair Plots" in eda_options and len(st.session_state['numeric_cols']) > 0:
                    st.subheader("Pair Plots (first 5 numeric features)")
                    pairplot_cols = st.session_state['numeric_cols'][:5]
                    if st.session_state['target_col'] in st.session_state['df'].columns:
                        pairplot_cols.append(st.session_state['target_col'])
                    fig = sns.pairplot(st.session_state['df'][pairplot_cols], 
                                      hue=st.session_state['target_col'] if st.session_state['target_col'] in st.session_state['df'].columns else None)
                    st.pyplot(fig)
                
                if "Categorical Analysis" in eda_options and len(st.session_state['categorical_cols']) > 0:
                    st.subheader("Categorical Features Analysis")
                    cat_cols = st.session_state['categorical_cols']
                    n_cols = 2
                    n_rows = (len(cat_cols)) // n_cols + 1
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*5))
                    axes = axes.flatten()
                    
                    for i, col in enumerate(cat_cols):
                        if col != st.session_state['target_col']:
                            sns.countplot(data=st.session_state['df'], x=col, 
                                         hue=st.session_state['target_col'] if st.session_state['target_col'] in st.session_state['df'].columns else None,
                                         ax=axes[i])
                            axes[i].tick_params(axis='x', rotation=45)
                            axes[i].set_title(col)
                    
                    for j in range(i+1, len(axes)):
                        axes[j].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                if "Missing Values" in eda_options:
                    st.subheader("Missing Values Analysis")
                    missing = st.session_state['df'].isnull().sum()
                    missing = missing[missing > 0]
                    if len(missing) > 0:
                        fig, ax = plt.subplots()
                        missing.plot(kind='bar', ax=ax)
                        st.pyplot(fig)
                    else:
                        st.info("No missing values found in the dataset.")
                
            except Exception as e:
                st.error(f"Error generating EDA: {str(e)}")
    
    # Data Preprocessing
    st.header("⚙️ Data Preprocessing")
    
    with st.expander("Data Imputation"):
        impute_strategy_num = st.selectbox(
            "Numeric imputation strategy:",
            ["mean", "median", "most_frequent", "constant"]
        )
        impute_strategy_cat = st.selectbox(
            "Categorical imputation strategy:",
            ["most_frequent", "constant"]
        )
    
    with st.expander("Feature Scaling"):
        scaling_method = st.selectbox(
            "Select scaling method:",
            ["Standard Scaler", "MinMax Scaler", "None"]
        )
    
    with st.expander("Feature Encoding"):
        encoding_method = st.selectbox(
            "Select encoding method:",
            ["One-Hot Encoding", "Ordinal Encoding"]
        )
    
    if st.button("Preprocess Data"):
        with st.spinner("Preprocessing data..."):
            try:
                # Separate features and target
                X = st.session_state['df'][st.session_state['selected_features']]
                y = st.session_state['df'][st.session_state['target_col']]
                
                # Define preprocessing steps
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=impute_strategy_num)),
                    ('scaler', StandardScaler() if scaling_method == "Standard Scaler" else 
                              MinMaxScaler() if scaling_method == "MinMax Scaler" else 'passthrough')
                ])
                
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=impute_strategy_cat, fill_value='missing')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore') if encoding_method == "One-Hot Encoding" else 'passthrough')
                ])
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, st.session_state['numeric_cols']),
                        ('cat', categorical_transformer, st.session_state['categorical_cols'])
                    ])
                
                # Apply preprocessing
                X_processed = preprocessor.fit_transform(X)
                
                # Get feature names after one-hot encoding
                if encoding_method == "One-Hot Encoding":
                    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
                    cat_features = cat_encoder.get_feature_names_out(st.session_state['categorical_cols'])
                    all_features = np.concatenate([st.session_state['numeric_cols'], cat_features])
                else:
                    all_features = st.session_state['selected_features']
                
                st.session_state['X_processed'] = X_processed
                st.session_state['y'] = y
                st.session_state['all_features'] = all_features
                st.session_state['preprocessor'] = preprocessor
                
                st.success("Data preprocessing completed successfully!")
                
                # Show processed data
                if encoding_method == "One-Hot Encoding":
                    processed_df = pd.DataFrame(X_processed.toarray(), columns=all_features)
                else:
                    processed_df = pd.DataFrame(X_processed, columns=all_features)
                
                with st.expander("View Processed Data"):
                    st.dataframe(processed_df.head())
                
            except Exception as e:
                st.error(f"Error during preprocessing: {str(e)}")
    
    # Feature Selection
    if 'X_processed' in st.session_state and 'y' in st.session_state:
        st.subheader("🔍 Feature Selection")
        
        if feature_selection_method != "None":
            with st.spinner("Performing feature selection..."):
                try:
                    X_processed = st.session_state['X_processed']
                    y = st.session_state['y']
                    all_features = st.session_state['all_features']
                    
                    if feature_selection_method == "SelectKBest":
                        selector = SelectKBest(score_func=f_classif, k=n_features)
                        X_selected = selector.fit_transform(X_processed, y)
                        selected_mask = selector.get_support()
                        selected_features = all_features[selected_mask]
                        
                        st.write(f"Selected {n_features} best features:")
                        st.write(selected_features)
                        
                        # Plot feature importance
                        fig, ax = plt.subplots()
                        pd.Series(selector.scores_, index=all_features).sort_values().plot(kind='barh', ax=ax)
                        ax.set_title("Feature Importance Scores")
                        st.pyplot(fig)
                    
                    elif feature_selection_method == "RFE":
                        estimator = RandomForestClassifier(random_state=random_state)
                        selector = RFE(estimator, n_features_to_select=n_features)
                        X_selected = selector.fit_transform(X_processed, y)
                        selected_mask = selector.get_support()
                        selected_features = all_features[selected_mask]
                        
                        st.write(f"Selected {n_features} features using RFE:")
                        st.write(selected_features)
                    
                    elif feature_selection_method == "Correlation Threshold":
                        if isinstance(X_processed, np.ndarray):
                            X_df = pd.DataFrame(X_processed, columns=all_features)
                        else:
                            X_df = pd.DataFrame(X_processed.toarray(), columns=all_features)
                        
                        # Calculate correlation with target
                        if isinstance(y, pd.Series):
                            y_encoded = y.astype('category').cat.codes
                        else:
                            y_encoded = y
                        
                        correlations = X_df.corrwith(pd.Series(y_encoded)).abs().sort_values(ascending=False)
                        selected_features = correlations.head(n_features).index.tolist()
                        
                        st.write(f"Top {n_features} correlated features:")
                        st.write(selected_features)
                        
                        # Plot correlations
                        fig, ax = plt.subplots()
                        correlations.head(20).plot(kind='bar', ax=ax)
                        ax.set_title("Feature Correlation with Target")
                        st.pyplot(fig)
                    
                    # Update processed data with selected features
                    if feature_selection_method != "Correlation Threshold":
                        if isinstance(X_processed, np.ndarray):
                            X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
                        else:
                            X_selected_df = pd.DataFrame(X_selected.toarray(), columns=selected_features)
                    else:
                        X_selected_df = X_df[selected_features]
                    
                    st.session_state['X_processed'] = X_selected_df.values
                    st.session_state['all_features'] = selected_features
                    st.session_state['selected_features'] = selected_features
                    
                except Exception as e:
                    st.error(f"Error during feature selection: {str(e)}")
    
    
 

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.stop()


## Training part

# Continue from Part 1

# Data Partitioning and Synthetic Data Generation
st.header("📊 Data Partitioning")
if 'X_processed' in st.session_state and 'y' in st.session_state:
    if st.button("Split Data and Generate Synthetic Samples"):
        with st.spinner("Partitioning data and generating synthetic samples..."):
            try:
                X = st.session_state['X_processed']
                y = st.session_state['y']
                
                # Initial train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=random_state, stratify=y
                )
                
                # Generate synthetic samples if needed
                if st.checkbox("Generate synthetic samples (SMOTE)"):
                    smote = SMOTE(random_state=random_state)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    st.success(f"Generated synthetic samples. New training size: {len(X_train)}")
                
                # Store in session state
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                
                # Display class distribution
                st.subheader("Class Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Training set:")
                    st.write(pd.Series(y_train).value_counts())
                
                with col2:
                    st.write("Test set:")
                    st.write(pd.Series(y_test).value_counts())
                
                # Cross-validation setup
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                st.session_state['cv'] = cv
                
                st.success("Data partitioning completed successfully!")
                
            except Exception as e:
                st.error(f"Error during data partitioning: {str(e)}")

# Model Training
st.header("🤖 Model Training")
if 'X_train' in st.session_state and 'y_train' in st.session_state:
    models = {
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "Multi-layer Perceptron": MLPClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(random_state=random_state, eval_metric='logloss'),
        "Explainable Boosting Classifier": ExplainableBoostingClassifier(random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state)
    }
    
    selected_models = st.multiselect(
        "Select models to train:",
        list(models.keys()),
        default=["Random Forest", "Logistic Regression"]
    )
    
    if st.button("Train Selected Models"):
        with st.spinner("Training models..."):
            try:
                model_results = {}
                feature_importances = {}
                
                for model_name in selected_models:
                    st.subheader(f"Training {model_name}")
                    
                    # Train model
                    model = models[model_name]
                    model.fit(st.session_state['X_train'], st.session_state['y_train'])
                    
                    # Make predictions
                    y_pred = model.predict(st.session_state['X_test'])
                    y_pred_proba = model.predict_proba(st.session_state['X_test'])[:, 1]
                    
                    # Store results
                    model_results[model_name] = {
                        'model': model,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba,
                        'y_test': st.session_state['y_test']
                    }
                    
                    # Get feature importances if available
                    if hasattr(model, 'feature_importances_'):
                        feature_importances[model_name] = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        feature_importances[model_name] = np.abs(model.coef_[0])
                    
                    # Display training results
                    st.write(f"{model_name} trained successfully!")
                
                st.session_state['model_results'] = model_results
                st.session_state['feature_importances'] = feature_importances
                st.success("All selected models trained successfully!")
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

# Hyperparameter Tuning
st.header("🎛️ Hyperparameter Tuning")
if 'X_train' in st.session_state and 'model_results' in st.session_state:
    tuning_method = st.selectbox(
        "Select tuning method:",
        ["Randomized Search", "Grid Search", "Bayesian Optimization"]
    )
    
    model_to_tune = st.selectbox(
        "Select model to tune:",
        list(st.session_state['model_results'].keys())
    )
    
    if st.button("Perform Hyperparameter Tuning"):
        with st.spinner("Running hyperparameter tuning..."):
            try:
                # Define parameter grids for different models
                param_grids = {
                    "Random Forest": {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    "Logistic Regression": {
                        'C': [0.001, 0.01, 0.1, 1, 10, 100],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'saga']
                    },
                    "Multi-layer Perceptron": {
                        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                        'activation': ['tanh', 'relu'],
                        'alpha': [0.0001, 0.001, 0.01]
                    },
                    "XGBoost": {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.6, 0.8, 1.0]
                    },
                    "Decision Tree": {
                        'max_depth': [None, 5, 10, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                }
                
                # Get the base model
                base_model = st.session_state['model_results'][model_to_tune]['model']
                
                # Perform tuning based on selected method
                if tuning_method == "Randomized Search":
                    tuner = RandomizedSearchCV(
                        estimator=base_model,
                        param_distributions=param_grids[model_to_tune],
                        n_iter=10,
                        cv=st.session_state['cv'],
                        random_state=random_state,
                        n_jobs=-1
                    )
                elif tuning_method == "Grid Search":
                    tuner = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grids[model_to_tune],
                        cv=st.session_state['cv'],
                        n_jobs=-1
                    )
                elif tuning_method == "Bayesian Optimization":
                    tuner = BayesSearchCV(
                        estimator=base_model,
                        search_spaces=param_grids[model_to_tune],
                        cv=st.session_state['cv'],
                        n_iter=10,
                        random_state=random_state,
                        n_jobs=-1
                    )
                
                tuner.fit(st.session_state['X_train'], st.session_state['y_train'])
                
                # Update the model with best parameters
                best_model = tuner.best_estimator_
                y_pred = best_model.predict(st.session_state['X_test'])
                y_pred_proba = best_model.predict_proba(st.session_state['X_test'])[:, 1]
                
                # Update results
                st.session_state['model_results'][model_to_tune] = {
                    'model': best_model,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'y_test': st.session_state['y_test']
                }
                
                st.success(f"Best parameters for {model_to_tune}:")
                st.write(tuner.best_params_)
                
            except Exception as e:
                st.error(f"Error during hyperparameter tuning: {str(e)}")

# Model Evaluation
st.header("📈 Model Evaluation")
if 'model_results' in st.session_state:
    # Confusion Matrix and Classification Report
    st.subheader("Confusion Matrix and Classification Metrics")
    
    selected_model = st.selectbox(
        "Select model to evaluate:",
        list(st.session_state['model_results'].keys())
    )
    
    if st.button("Evaluate Selected Model"):
        with st.spinner("Generating evaluation metrics..."):
            try:
                results = st.session_state['model_results'][selected_model]
                y_test = results['y_test']
                y_pred = results['y_pred']
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
    
    # ROC Curve Comparison
    st.subheader("ROC Curve Comparison")
    if st.button("Plot ROC Curves for All Models"):
        with st.spinner("Generating ROC curves..."):
            try:
                fig, ax = plt.subplots()
                
                for model_name, results in st.session_state['model_results'].items():
                    y_test = results['y_test']
                    y_pred_proba = results['y_pred_proba']
                    
                    if isinstance(y_test, pd.Series):
                        y_test = y_test.astype('category').cat.codes
                    
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
                
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc="lower right")
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error generating ROC curves: {str(e)}")

# Model Interpretation
st.header("🔍 Model Interpretation")
if 'model_results' in st.session_state and 'all_features' in st.session_state:
    # Feature Importance
    st.subheader("Feature Importance")
    
    if 'feature_importances' in st.session_state:
        importance_model = st.selectbox(
            "Select model for feature importance:",
            list(st.session_state['feature_importances'].keys())
        )
        
        if st.button("Plot Feature Importance"):
            with st.spinner("Generating feature importance plot..."):
                try:
                    importances = st.session_state['feature_importances'][importance_model]
                    features = st.session_state['all_features']
                    
                    # Create DataFrame for visualization
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                    ax.set_title(f'Feature Importance - {importance_model}')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error plotting feature importance: {str(e)}")
    
    # SHAP Values
    st.subheader("SHAP Values")
    shap_model = st.selectbox(
        "Select model for SHAP analysis:",
        [m for m in st.session_state['model_results'].keys() if m not in ["Logistic Regression", "Multi-layer Perceptron"]]
    )
    
    if st.button("Generate SHAP Summary"):
        with st.spinner("Calculating SHAP values..."):
            try:
                model = st.session_state['model_results'][shap_model]['model']
                X_train = st.session_state['X_train']
                feature_names = st.session_state['all_features']
                
                # Convert to DataFrame if not already
                if isinstance(X_train, np.ndarray):
                    X_train_df = pd.DataFrame(X_train, columns=feature_names)
                else:
                    X_train_df = pd.DataFrame(X_train.toarray(), columns=feature_names)
                
                # Initialize SHAP explainer based on model type
                if shap_model == "Random Forest" or shap_model == "Decision Tree":
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_train_df)
                elif shap_model == "XGBoost":
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X_train_df)
                elif shap_model == "Explainable Boosting Classifier":
                    explainer = shap.Explainer(model.predict, X_train_df)
                    shap_values = explainer(X_train_df)
                
                # Plot summary
                st.set_option('deprecation.showPyplotGlobalUse', False)
                fig, ax = plt.subplots()
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values[1], X_train_df, plot_type="bar", show=False)
                else:
                    shap.summary_plot(shap_values, X_train_df, plot_type="bar", show=False)
                st.pyplot(fig)
                
                # Force plot for a single instance
                st.subheader("SHAP Force Plot for Single Instance")
                instance_idx = st.slider("Select instance index", 0, len(X_train_df)-1, 0)
                
                fig = plt.figure()
                if isinstance(shap_values, list):
                    shap.force_plot(explainer.expected_value[1], shap_values[1][instance_idx,:], 
                                   X_train_df.iloc[instance_idx,:], matplotlib=True, show=False)
                else:
                    shap.force_plot(explainer.expected_value, shap_values[instance_idx,:], 
                                   X_train_df.iloc[instance_idx,:], matplotlib=True, show=False)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error generating SHAP values: {str(e)}")
    
    # LIME Explanations
    st.subheader("LIME Explanations")
    lime_model = st.selectbox(
        "Select model for LIME explanation:",
        list(st.session_state['model_results'].keys())
    )
    
    if st.button("Generate LIME Explanation"):
        with st.spinner("Generating LIME explanation..."):
            try:
                model = st.session_state['model_results'][lime_model]['model']
                X_train = st.session_state['X_train']
                feature_names = st.session_state['all_features']
                
                # Convert to DataFrame if not already
                if isinstance(X_train, np.ndarray):
                    X_train_df = pd.DataFrame(X_train, columns=feature_names)
                else:
                    X_train_df = pd.DataFrame(X_train.toarray(), columns=feature_names)
                
                # Initialize LIME explainer
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train_df.values,
                    feature_names=feature_names,
                    class_names=['No Attrition', 'Attrition'],
                    verbose=True,
                    mode='classification'
                )
                
                # Select instance
                instance_idx = st.slider("Select instance index for LIME", 0, len(X_train_df)-1, 0)
                instance = X_train_df.iloc[instance_idx].values
                
                # Explain instance
                exp = explainer.explain_instance(
                    instance,
                    model.predict_proba,
                    num_features=min(5, len(feature_names))
                
                # Display explanation
                st.write(exp.as_list())
                
                # Plot explanation
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error generating LIME explanation: {str(e)}")
    
    # Decision Rules (for tree-based models)
    st.subheader("Decision Rules")
    rules_model = st.selectbox(
        "Select tree-based model for decision rules:",
        [m for m in st.session_state['model_results'].keys() if m in ["Random Forest", "Decision Tree"]]
    )
    
    if st.button("Extract Decision Rules"):
        with st.spinner("Extracting decision rules..."):
            try:
                model = st.session_state['model_results'][rules_model]['model']
                feature_names = st.session_state['all_features']
                
                if rules_model == "Decision Tree":
                    tree_rules = export_text(
                        model,
                        feature_names=list(feature_names))
                    st.text(tree_rules)
                elif rules_model == "Random Forest":
                    # Get one tree from the forest
                    estimator = model.estimators_[0]
                    tree_rules = export_text(
                        estimator,
                        feature_names=list(feature_names))
                    st.text(tree_rules)
                    st.info("Showing rules for one tree in the random forest.")
                
            except Exception as e:
                st.error(f"Error extracting decision rules: {str(e)}")

# Final Recommendations
st.header("🎯 Recommendations")
if 'model_results' in st.session_state:
    if st.button("Generate Attrition Prevention Recommendations"):
        with st.spinner("Generating recommendations..."):
            try:
                # Get feature importances from the best model
                best_model_name = max(
                    st.session_state['model_results'].keys(),
                    key=lambda x: roc_auc_score(
                        st.session_state['model_results'][x]['y_test'],
                        st.session_state['model_results'][x]['y_pred_proba']
                    )
                )
                
                # Get top features
                if 'feature_importances' in st.session_state and best_model_name in st.session_state['feature_importances']:
                    importances = st.session_state['feature_importances'][best_model_name]
                    features = st.session_state['all_features']
                    
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(5)
                    
                    top_features = importance_df['Feature'].tolist()
                    
                    # Generate recommendations based on top features
                    recommendations = []
                    
                    if 'MonthlyIncome' in top_features:
                        recommendations.append(
                            "💵 **Compensation Review**: Employees with lower monthly income are more likely to leave. "
                            "Consider reviewing compensation packages to ensure they are competitive with market rates."
                        )
                    
                    if 'YearsAtCompany' in top_features:
                        recommendations.append(
                            "📅 **Career Progression**: Employees who have been with the company longer may feel stagnant. "
                            "Implement clear career progression paths and regular promotion cycles."
                        )
                    
                    if 'WorkLifeBalance' in top_features:
                        recommendations.append(
                            "⚖️ **Work-Life Balance**: Poor work-life balance is a significant factor. "
                            "Consider flexible working hours, remote work options, and workload assessments."
                        )
                    
                    if 'JobSatisfaction' in top_features:
                        recommendations.append(
                            "😊 **Job Satisfaction**: Low job satisfaction correlates with attrition. "
                            "Conduct regular employee satisfaction surveys and address key concerns."
                        )
                    
                    if 'YearsSinceLastPromotion' in top_features:
                        recommendations.append(
                            "📈 **Promotion Frequency**: Employees who haven't been promoted in a while are at risk. "
                            "Ensure regular performance reviews and promotion opportunities."
                        )
                    
                    # If no specific features matched, provide general recommendations
                    if not recommendations:
                        recommendations.append(
                            "🔍 Based on the model, the top factors driving attrition are: " + 
                            ", ".join(top_features) + ". Focus interventions on these areas."
                        )
                    
                    # Display recommendations
                    st.subheader(f"Top 5 Features from {best_model_name}:")
                    st.write(importance_df)
                    
                    st.subheader("Recommended Actions to Reduce Attrition:")
                    for rec in recommendations:
                        st.info(rec)
                
                else:
                    st.warning("Could not generate specific recommendations without feature importances.")
                
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")

# Download Results
st.header("📥 Download Results")
if 'model_results' in st.session_state:
    # Create a report DataFrame
    report_data = []
    
    for model_name, results in st.session_state['model_results'].items():
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='Yes' if 'Yes' in y_test.values else 1)
        recall = recall_score(y_test, y_pred, pos_label='Yes' if 'Yes' in y_test.values else 1)
        f1 = f1_score(y_test, y_pred, pos_label='Yes' if 'Yes' in y_test.values else 1)
        roc_auc = roc_auc_score(y_test, results['y_pred_proba'])
        
        report_data.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })
    
    report_df = pd.DataFrame(report_data)
    
    # Convert to CSV
    csv = report_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Model Performance Report",
        data=csv,
        file_name='attrition_model_performance.csv',
        mime='text/csv')