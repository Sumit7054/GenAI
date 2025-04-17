import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pandasai import SmartDataframe
import warnings
from langchain_community.llms import Ollama
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve
import lime
from lime import lime_tabular

# Suppress warnings
warnings.filterwarnings('ignore')

llm = Ollama(model="mistral")

# Set page config
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'X_processed' not in st.session_state:
    st.session_state.X_processed = None
if 'y' not in st.session_state:
    st.session_state.y = None

# Title and description
st.title("👥 Employee Attrition Prediction")
st.markdown("""
This application helps predict employee attrition using machine learning. 
Upload your dataset or use the default one, then follow the steps to analyze and predict attrition.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
steps = [
    "1. Data Loading & Selection",
    "2. Data Cleaning",
    "3. Exploratory Data Analysis",
    "4. Data Preprocessing",
    "5. Feature Engineering",
    "6. Model Training",
    "7. Hyperparameter Tuning",
    "8. Model Evaluation",
    "9. Explainability"
]
current_step = st.sidebar.radio("Go to", steps)

# Step 1: Data Loading & Selection
if current_step == steps[0]:
    st.header("1. Data Loading & Selection")
    
    data_option = st.radio("Choose data source:", 
                         ["Use default dataset (IBM HR Analytics)", "Upload your own CSV file"])
    
    if data_option == "Use default dataset (IBM HR Analytics)":
        try:
            # Load default dataset
            df = pd.read_csv("https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv")
            st.session_state.df = df
            st.success("Default dataset loaded successfully!")
            
            # Set default target and features
            st.session_state.target = "Attrition"
            default_features = [col for col in df.columns if col != "Attrition"]
            st.session_state.features = default_features
            
            st.write("Preview of the dataset:")
            st.dataframe(df.head())
            
            st.info(f"Default target variable set to: 'Attrition'")
            st.info(f"Default features selected: All columns except 'Attrition'")
            
        except Exception as e:
            st.error(f"Error loading default dataset: {str(e)}")
    else:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success("File uploaded successfully!")
                
                st.write("Preview of the dataset:")
                st.dataframe(df.head())
                
                # Let user select target and features
                all_columns = df.columns.tolist()
                target = st.selectbox("Select the target variable (what you want to predict):", all_columns)
                st.session_state.target = target
                
                features = st.multiselect("Select features to use for prediction:", 
                                         [col for col in all_columns if col != target],
                                         default=[col for col in all_columns if col != target])
                st.session_state.features = features
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

# Step 2: Data Cleaning
elif current_step == steps[1] and st.session_state.df is not None:
    st.header("2. Data Cleaning")
    df = st.session_state.df
    
    st.subheader("Current Data Summary")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        st.success("No missing values found in the dataset!")
    else:
        st.warning(f"Found {missing_values.sum()} missing values in the dataset.")
        st.dataframe(missing_values[missing_values > 0].rename("Missing Values"))
        
        # Missing value treatment options
        st.subheader("Missing Value Treatment")
        treatment_option = st.selectbox(
            "How would you like to handle missing values?",
            ["Drop rows with missing values", 
             "Fill with mean (numeric) / mode (categorical)",
             "Fill with median (numeric) / mode (categorical)",
             "Fill with constant value",
             "Use LLM to suggest treatment"]
        )
        
        if st.button("Apply Missing Value Treatment"):
            if treatment_option == "Drop rows with missing values":
                df = df.dropna()
                st.success(f"Dropped rows with missing values. New shape: {df.shape}")
            elif treatment_option == "Fill with mean (numeric) / mode (categorical)":
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
                st.success("Filled missing values with mean (numeric) / mode (categorical)")
            elif treatment_option == "Fill with median (numeric) / mode (categorical)":
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
                st.success("Filled missing values with median (numeric) / mode (categorical)")
            elif treatment_option == "Fill with constant value":
                constant_value = st.text_input("Enter constant value to fill (e.g., 0, 'Unknown')", "Unknown")
                df = df.fillna(constant_value)
                st.success(f"Filled missing values with constant: {constant_value}")
            elif treatment_option == "Use LLM to suggest treatment":
                try:
                    # Using pandas-ai to suggest treatment
                    sdf = SmartDataframe(df, config={"llm": llm})
                    suggestion = sdf.chat("How should I handle missing values in this dataset?")
                    st.info(f"LLM Suggestion: {suggestion}")
                except Exception as e:
                    st.error(f"Error getting LLM suggestion: {str(e)}")
            
            st.session_state.df = df
            st.write("Updated missing values count:")
            st.write(df.isnull().sum())
    
    # Duplicate values analysis
    st.subheader("Duplicate Values Analysis")
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        st.success("No duplicate rows found!")
    else:
        st.warning(f"Found {duplicates} duplicate rows in the dataset.")
        if st.button("Remove Duplicate Rows"):
            df = df.drop_duplicates()
            st.session_state.df = df
            st.success(f"Removed duplicate rows. New shape: {df.shape}")
    
    # Outlier detection (for numeric columns)
    st.subheader("Outlier Detection")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        selected_num_col = st.selectbox("Select numeric column for outlier detection:", numeric_cols)
        
        # Boxplot for outlier visualization
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_num_col], ax=ax)
        st.pyplot(fig)
        
        # Outlier treatment options
        st.subheader("Outlier Treatment")
        outlier_option = st.selectbox(
            "How would you like to handle outliers?",
            ["No treatment", 
             "Remove outliers (IQR method)",
             "Cap outliers (IQR method)",
             "Use LLM to suggest treatment"]
        )
        
        if outlier_option != "No treatment" and st.button("Apply Outlier Treatment"):
            if outlier_option == "Remove outliers (IQR method)":
                Q1 = df[selected_num_col].quantile(0.25)
                Q3 = df[selected_num_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[selected_num_col] >= lower_bound) & (df[selected_num_col] <= upper_bound)]
                st.session_state.df = df
                st.success(f"Removed outliers. New shape: {df.shape}")
            elif outlier_option == "Cap outliers (IQR method)":
                Q1 = df[selected_num_col].quantile(0.25)
                Q3 = df[selected_num_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[selected_num_col] = np.where(df[selected_num_col] < lower_bound, lower_bound, 
                                             np.where(df[selected_num_col] > upper_bound, upper_bound, 
                                                     df[selected_num_col]))
                st.session_state.df = df
                st.success("Capped outliers using IQR method")
            elif outlier_option == "Use LLM to suggest treatment":
                try:
                    # Using pandas-ai to suggest treatment
                    sdf = SmartDataframe(df, config={"llm": llm})
                    suggestion = sdf.chat(f"How should I handle outliers in column {selected_num_col}?")
                    st.info(f"LLM Suggestion: {suggestion}")
                except Exception as e:
                    st.error(f"Error getting LLM suggestion: {str(e)}")
    else:
        st.info("No numeric columns found for outlier detection.")
    
    # Data type conversion
    st.subheader("Data Type Conversion")
    st.write("Current data types:")
    st.write(df.dtypes)
    
    convert_col = st.selectbox("Select column to convert type:", df.columns)
    new_type = st.selectbox("Select new data type:", 
                           ["object", "int64", "float64", "datetime64", "category", "bool"])
    
    if st.button("Convert Data Type"):
        try:
            if new_type == "datetime64":
                date_format = st.text_input("Enter date format (e.g., %Y-%m-%d)", "%Y-%m-%d")
                df[convert_col] = pd.to_datetime(df[convert_col], format=date_format)
            elif new_type == "category":
                df[convert_col] = df[convert_col].astype('category')
            else:
                df[convert_col] = df[convert_col].astype(new_type)
            st.session_state.df = df
            st.success(f"Converted {convert_col} to {new_type}")
            st.write("Updated data types:")
            st.write(df.dtypes)
        except Exception as e:
            st.error(f"Error converting type: {str(e)}")

# Step 3: Exploratory Data Analysis
elif current_step == steps[2] and st.session_state.df is not None:
    st.header("3. Exploratory Data Analysis")
    df = st.session_state.df
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))
    
    # Target variable distribution
    if st.session_state.target is not None:
        st.subheader(f"Target Variable Distribution: {st.session_state.target}")
        fig, ax = plt.subplots()
        if df[st.session_state.target].nunique() <= 10:
            df[st.session_state.target].value_counts().plot(kind='bar', ax=ax)
        else:
            sns.histplot(df[st.session_state.target], kde=True, ax=ax)
        st.pyplot(fig)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Top correlated features with target (if numeric)
        if st.session_state.target in numeric_cols:
            st.write("Top features correlated with target:")
            target_corr = corr_matrix[st.session_state.target].sort_values(ascending=False)
            st.write(target_corr)
    else:
        st.info("Not enough numeric columns for correlation analysis.")
    
    # Feature distributions
    st.subheader("Feature Distributions")
    selected_col = st.selectbox("Select column to visualize:", df.columns)
    
    if df[selected_col].dtype in ['int64', 'float64']:
        plot_type = st.radio("Select plot type:", ["Histogram", "Boxplot", "Violin Plot"])
        fig, ax = plt.subplots()
        if plot_type == "Histogram":
            sns.histplot(df[selected_col], kde=True, ax=ax)
        elif plot_type == "Boxplot":
            sns.boxplot(x=df[selected_col], ax=ax)
        elif plot_type == "Violin Plot":
            sns.violinplot(x=df[selected_col], ax=ax)
        st.pyplot(fig)
    else:
        # For categorical columns
        plot_type = st.radio("Select plot type:", ["Bar Chart", "Pie Chart"])
        fig, ax = plt.subplots()
        if plot_type == "Bar Chart":
            df[selected_col].value_counts().plot(kind='bar', ax=ax)
        elif plot_type == "Pie Chart":
            df[selected_col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
    
    # Bivariate analysis
    if st.session_state.target is not None:
        st.subheader(f"Bivariate Analysis with Target: {st.session_state.target}")
        feature_for_bivariate = st.selectbox("Select feature for bivariate analysis:", 
                                           [col for col in df.columns if col != st.session_state.target])
        
        fig, ax = plt.subplots()
        if df[st.session_state.target].nunique() <= 10 and df[feature_for_bivariate].dtype in ['int64', 'float64']:
            # Boxplot for categorical target vs numeric feature
            sns.boxplot(x=df[st.session_state.target], y=df[feature_for_bivariate], ax=ax)
        elif df[st.session_state.target].nunique() <= 10 and df[feature_for_bivariate].dtype == 'object':
            # Countplot for categorical vs categorical
            sns.countplot(x=feature_for_bivariate, hue=st.session_state.target, data=df, ax=ax)
            plt.xticks(rotation=45)
        elif df[st.session_state.target].dtype in ['int64', 'float64'] and df[feature_for_bivariate].dtype in ['int64', 'float64']:
            # Scatterplot for numeric vs numeric
            sns.scatterplot(x=feature_for_bivariate, y=st.session_state.target, data=df, ax=ax)
        st.pyplot(fig)
    
    # Use LLM for EDA insights
    if st.button("Get LLM Insights on Data"):
        try:
            sdf = SmartDataframe(df, config={"llm": llm})
            insights = sdf.chat("What are the key insights from this dataset for predicting employee attrition?")
            st.subheader("LLM Generated Insights")
            st.write(insights)
        except Exception as e:
            st.error(f"Error getting LLM insights: {str(e)}")

# Step 4: Data Preprocessing
elif current_step == steps[3] and st.session_state.df is not None and st.session_state.target is not None:
    st.header("4. Data Preprocessing")
    df = st.session_state.df
    target = st.session_state.target
    features = st.session_state.features
    
    # Separate features and target
    X = df[features]
    y = df[target]
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.subheader("Current Features Summary")
    st.write(f"Numeric features: {numeric_cols}")
    st.write(f"Categorical features: {categorical_cols}")
    
    # Preprocessing options
    st.subheader("Preprocessing Options")
    
    # Numeric preprocessing
    st.markdown("**Numeric Features Processing**")
    numeric_scaling = st.selectbox("Scaling method for numeric features:",
                                 ["None", "Standard Scaler (mean=0, std=1)", 
                                  "MinMax Scaler (0-1)", "Robust Scaler (median/IQR)"])
    
    numeric_impute = st.selectbox("Imputation method for numeric features:",
                                ["None", "Mean", "Median", "Constant (specify)"])
    
    if numeric_impute == "Constant (specify)":
        numeric_constant = st.number_input("Constant value for numeric imputation:", value=0)
    
    # Categorical preprocessing
    st.markdown("**Categorical Features Processing**")
    categorical_encode = st.selectbox("Encoding method for categorical features:",
                                    ["None", "One-Hot Encoding", "Ordinal Encoding"])
    
    categorical_impute = st.selectbox("Imputation method for categorical features:",
                                    ["None", "Most Frequent", "Constant (specify)"])
    
    if categorical_impute == "Constant (specify)":
        categorical_constant = st.text_input("Constant value for categorical imputation:", "Missing")
    
    # Apply preprocessing
    if st.button("Apply Preprocessing"):
        try:
            # Create transformers
            numeric_transforms = []
            if numeric_impute != "None":
                if numeric_impute == "Mean":
                    numeric_transforms.append(('imputer', SimpleImputer(strategy='mean')))
                elif numeric_impute == "Median":
                    numeric_transforms.append(('imputer', SimpleImputer(strategy='median')))
                elif numeric_impute == "Constant (specify)":
                    numeric_transforms.append(('imputer', SimpleImputer(strategy='constant', fill_value=numeric_constant)))
            
            if numeric_scaling != "None":
                if numeric_scaling == "Standard Scaler (mean=0, std=1)":
                    numeric_transforms.append(('scaler', StandardScaler()))
                elif numeric_scaling == "MinMax Scaler (0-1)":
                    from sklearn.preprocessing import MinMaxScaler
                    numeric_transforms.append(('scaler', MinMaxScaler()))
                elif numeric_scaling == "Robust Scaler (median/IQR)":
                    from sklearn.preprocessing import RobustScaler
                    numeric_transforms.append(('scaler', RobustScaler()))
            
            categorical_transforms = []
            if categorical_impute != "None":
                if categorical_impute == "Most Frequent":
                    categorical_transforms.append(('imputer', SimpleImputer(strategy='most_frequent')))
                elif categorical_impute == "Constant (specify)":
                    categorical_transforms.append(('imputer', SimpleImputer(strategy='constant', fill_value=categorical_constant)))
            
            if categorical_encode != "None":
                if categorical_encode == "One-Hot Encoding":
                    categorical_transforms.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
                elif categorical_encode == "Ordinal Encoding":
                    from sklearn.preprocessing import OrdinalEncoder
                    categorical_transforms.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
            
            # Create column transformer
            transformers = []
            if numeric_transforms:
                transformers.append(('num', Pipeline(numeric_transforms), numeric_cols))
            if categorical_transforms:
                transformers.append(('cat', Pipeline(categorical_transforms), categorical_cols))
            
            if transformers:
                preprocessor = ColumnTransformer(transformers, remainder='passthrough')
                
                # Fit and transform the data
                X_processed = preprocessor.fit_transform(X)
                
                # Get feature names after transformation
                feature_names = []
                if numeric_transforms:
                    feature_names.extend(numeric_cols)
                
                if categorical_transforms and categorical_encode == "One-Hot Encoding":
                    # Get one-hot encoded feature names
                    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
                    cat_feature_names = encoder.get_feature_names_out(categorical_cols)
                    feature_names.extend(cat_feature_names)
                elif categorical_transforms and categorical_encode == "Ordinal Encoding":
                    feature_names.extend(categorical_cols)
                
                # Handle remaining features
                remaining_cols = [col for col in X.columns if col not in numeric_cols + categorical_cols]
                feature_names.extend(remaining_cols)
                
                # Create DataFrame with processed features
                X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
                
                st.session_state.X_processed = X_processed_df
                st.session_state.y = y
                st.session_state.preprocessor = preprocessor
                
                st.success("Preprocessing applied successfully!")
                st.write("Processed features preview:")
                st.dataframe(X_processed_df.head())
                
                # Show shape info
                st.write(f"Original shape: {X.shape}")
                st.write(f"Processed shape: {X_processed_df.shape}")
            else:
                st.info("No preprocessing steps selected.")
                
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
            
# Step 5: Feature Engineering
elif current_step == steps[4] and hasattr(st.session_state, 'X_processed'):
    st.header("5. Feature Engineering")
    X_processed = st.session_state.X_processed
    y = st.session_state.y
    
    st.subheader("Current Features")
    st.write(X_processed.head())
    
    # Feature selection methods
    st.subheader("Feature Selection Methods")
    selection_method = st.selectbox("Select feature selection method:",
                                  ["None", 
                                   "SelectKBest (ANOVA F-value)", 
                                   "Recursive Feature Elimination (RFE)",
                                   "Feature Importance (Random Forest)",
                                   "Use LLM to suggest features"])
    
    if selection_method != "None" and st.button("Apply Feature Selection"):
        try:
            if selection_method == "SelectKBest (ANOVA F-value)":
                k = st.slider("Select number of top features to keep:", 
                             min_value=1, 
                             max_value=min(50, X_processed.shape[1]), 
                             value=min(10, X_processed.shape[1]))
                
                from sklearn.feature_selection import SelectKBest, f_classif
                selector = SelectKBest(score_func=f_classif, k=k)
                X_selected = selector.fit_transform(X_processed, y)
                
                # Get selected features
                selected_features = X_processed.columns[selector.get_support()]
                st.write(f"Selected {k} best features:")
                st.write(selected_features.tolist())
                
                st.session_state.X_selected = pd.DataFrame(X_selected, columns=selected_features)
                st.session_state.feature_selector = selector
                
            elif selection_method == "Recursive Feature Elimination (RFE)":
                n_features = st.slider("Select number of features to select:", 
                                     min_value=1, 
                                     max_value=min(50, X_processed.shape[1]), 
                                     value=min(10, X_processed.shape[1]))
                
                from sklearn.feature_selection import RFE
                from sklearn.linear_model import LogisticRegression
                estimator = LogisticRegression(max_iter=1000)
                selector = RFE(estimator, n_features_to_select=n_features)
                X_selected = selector.fit_transform(X_processed, y)
                
                # Get selected features
                selected_features = X_processed.columns[selector.support_]
                st.write(f"Selected {n_features} features using RFE:")
                st.write(selected_features.tolist())
                
                st.session_state.X_selected = pd.DataFrame(X_selected, columns=selected_features)
                st.session_state.feature_selector = selector
                
            elif selection_method == "Feature Importance (Random Forest)":
                threshold = st.slider("Select importance threshold:", 
                                    min_value=0.0, 
                                    max_value=1.0, 
                                    value=0.01, 
                                    step=0.01)
                
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_processed, y)
                
                importances = rf.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': X_processed.columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                st.write("Feature Importances:")
                st.dataframe(importance_df)
                
                # Plot feature importances
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                st.pyplot(fig)
                
                # Select features above threshold
                selected_features = importance_df[importance_df['Importance'] >= threshold]['Feature']
                X_selected = X_processed[selected_features]
                
                st.write(f"Selected {len(selected_features)} features with importance >= {threshold}:")
                st.write(selected_features.tolist())
                
                st.session_state.X_selected = X_selected
                st.session_state.feature_selector = rf
                
            elif selection_method == "Use LLM to suggest features":
                try:
                    # Combine X and y for LLM analysis
                    df_for_llm = pd.concat([X_processed, y], axis=1)
                    sdf = SmartDataframe(df_for_llm, config={"llm": llm})
                    
                    # Ask LLM for feature suggestions
                    response = sdf.chat("Which features are most important for predicting the target? Please list the top 10 most important features.")
                    
                    st.subheader("LLM Feature Selection Suggestions")
                    st.write(response)
                    
                    # Let user select features based on LLM suggestion
                    llm_selected_features = st.multiselect("Select features based on LLM suggestion:", 
                                                          X_processed.columns,
                                                          default=X_processed.columns.tolist()[:10])
                    
                    if llm_selected_features:
                        X_selected = X_processed[llm_selected_features]
                        st.session_state.X_selected = X_selected
                        st.success(f"Selected {len(llm_selected_features)} features based on LLM suggestion")
                        st.write(X_selected.head())
                    
                except Exception as e:
                    st.error(f"Error getting LLM feature suggestions: {str(e)}")
            
            if 'X_selected' in st.session_state:
                st.success("Feature selection completed!")
                st.write(f"Selected features shape: {st.session_state.X_selected.shape}")
                
        except Exception as e:
            st.error(f"Error during feature selection: {str(e)}")
    
    # Feature creation
    st.subheader("Feature Creation")
    st.markdown("""
    Create new features from existing ones. For example:
    - Interaction terms (feature1 * feature2)
    - Polynomial features
    - Binning numeric features
    - Date/time features from timestamps
    """)
    
    feature_create_option = st.selectbox("Select feature creation method:",
                                      ["None",
                                       "Interaction Terms",
                                       "Polynomial Features",
                                       "Binning Numeric Features",
                                       "Use LLM to suggest feature creation"])
    
    if feature_create_option != "None" and st.button("Create New Features"):
        try:
            X_to_use = st.session_state.X_selected if 'X_selected' in st.session_state else st.session_state.X_processed
            
            if feature_create_option == "Interaction Terms":
                col1, col2 = st.multiselect("Select two features for interaction term:", 
                                          X_to_use.columns, 
                                          max_selections=2)
                
                if len(col1) == 2:
                    new_feature_name = f"{col1[0]}_x_{col1[1]}"
                    X_to_use[new_feature_name] = X_to_use[col1[0]] * X_to_use[col1[1]]
                    st.success(f"Created interaction term: {new_feature_name}")
                    st.session_state.X_selected = X_to_use
                    st.write(X_to_use.head())
            
            elif feature_create_option == "Polynomial Features":
                degree = st.slider("Select polynomial degree:", 2, 5, 2)
                
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_poly = poly.fit_transform(X_to_use.select_dtypes(include=['int64', 'float64']))
                
                # Get feature names
                poly_feature_names = poly.get_feature_names_out(X_to_use.select_dtypes(include=['int64', 'float64']).columns)
                
                # Create DataFrame with polynomial features
                X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)
                
                # Combine with original features
                X_combined = pd.concat([X_to_use, X_poly_df], axis=1)
                
                st.session_state.X_selected = X_combined
                st.success(f"Created polynomial features (degree {degree})")
                st.write(X_combined.head())
            
            elif feature_create_option == "Binning Numeric Features":
                num_col = st.selectbox("Select numeric column to bin:", 
                                    X_to_use.select_dtypes(include=['int64', 'float64']).columns)
                
                bin_method = st.selectbox("Select binning method:", 
                                        ["Equal Width", 
                                         "Equal Frequency", 
                                         "Custom Bins"])
                
                n_bins = st.slider("Number of bins:", 2, 10, 4)
                
                if bin_method == "Equal Width":
                    X_to_use[f"{num_col}_binned"] = pd.cut(X_to_use[num_col], bins=n_bins, labels=False)
                elif bin_method == "Equal Frequency":
                    X_to_use[f"{num_col}_binned"] = pd.qcut(X_to_use[num_col], q=n_bins, labels=False)
                elif bin_method == "Custom Bins":
                    bin_edges = st.text_input("Enter bin edges (comma separated):", 
                                            "0,25,50,75,100")
                    bins = [float(x.strip()) for x in bin_edges.split(",")]
                    X_to_use[f"{num_col}_binned"] = pd.cut(X_to_use[num_col], bins=bins, labels=False)
                
                st.session_state.X_selected = X_to_use
                st.success(f"Created binned version of {num_col}")
                st.write(X_to_use.head())
            
            elif feature_create_option == "Use LLM to suggest feature creation":
                try:
                    # Combine X and y for LLM analysis
                    df_for_llm = pd.concat([X_to_use, y], axis=1)
                    sdf = SmartDataframe(df_for_llm, config={"llm": llm})
                    
                    # Ask LLM for feature creation suggestions
                    response = sdf.chat("What new features could be created from the existing ones that might help predict the target better?")
                    
                    st.subheader("LLM Feature Creation Suggestions")
                    st.write(response)
                    
                except Exception as e:
                    st.error(f"Error getting LLM feature creation suggestions: {str(e)}")
        
        except Exception as e:
            st.error(f"Error during feature creation: {str(e)}")

# Step 6: Model Training
elif current_step == steps[5] and hasattr(st.session_state, 'X_processed'):
    st.header("6. Model Training")
    
    # Get the data to use
    X_to_use = st.session_state.X_selected if 'X_selected' in st.session_state else st.session_state.X_processed
    y = st.session_state.y
    
    # Data partitioning
    st.subheader("Data Partitioning")
    test_size = st.slider("Test set size (%):", 10, 40, 20)
    random_state = st.number_input("Random state:", value=42)
    
    if st.button("Split Data into Train/Test Sets"):
        try:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_to_use, y, test_size=test_size/100, random_state=random_state, stratify=y
            )
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.success("Data split successfully!")
            st.write(f"Training set shape: {X_train.shape}")
            st.write(f"Test set shape: {X_test.shape}")
            
            # Show class distribution
            st.write("Class distribution in training set:")
            st.write(y_train.value_counts(normalize=True))
            st.write("Class distribution in test set:")
            st.write(y_test.value_counts(normalize=True))
            
        except Exception as e:
            st.error(f"Error splitting data: {str(e)}")
    
    # Model selection
    if 'X_train' in st.session_state:
        st.subheader("Model Selection")
        
        # Let user select which models to train
        model_options = {
            "Random Forest": RandomForestClassifier(random_state=random_state),
            "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
            "Multi-layer Perceptron": MLPClassifier(random_state=random_state),
            "XGBoost": GradientBoostingClassifier(random_state=random_state),
            "Explainable Boosting Classifier": ExplainableBoostingClassifier(random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state)
        }
        
        selected_models = st.multiselect("Select models to train:", list(model_options.keys()),
                                        default=["Random Forest", "Logistic Regression"])
        
        if st.button("Train Selected Models"):
            try:
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                st.session_state.models = {}
                results = []
                
                for model_name in selected_models:
                    with st.spinner(f"Training {model_name}..."):
                        model = model_options[model_name]
                        model.fit(X_train, y_train)
                        
                        # Store model
                        st.session_state.models[model_name] = model
                        
                        # Evaluate on test set
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                        # Store results
                        results.append({
                            "Model": model_name,
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1 Score": f1
                        })
                
                # Display results
                results_df = pd.DataFrame(results)
                st.subheader("Model Performance Comparison")
                st.dataframe(results_df.sort_values("F1 Score", ascending=False))
                
                # Store best model
                best_model_name = results_df.loc[results_df['F1 Score'].idxmax(), 'Model']
                st.session_state.best_model = st.session_state.models[best_model_name]
                st.success(f"Best model: {best_model_name} (F1 Score: {results_df['F1 Score'].max():.3f})")
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

# Step 7: Hyperparameter Tuning
elif current_step == steps[6] and 'models' in st.session_state and st.session_state.models:
    st.header("7. Hyperparameter Tuning")
    
    model_to_tune = st.selectbox("Select model to tune:", list(st.session_state.models.keys()))
    
    tuning_method = st.selectbox("Select tuning method:",
                               ["Randomized Search", 
                                "Grid Search", 
                                "Bayesian Optimization"])
    
    if st.button(f"Tune {model_to_tune}"):
        try:
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            
            # Define parameter grids for each model
            param_grids = {
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "bootstrap": [True, False]
                },
                "Logistic Regression": {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "penalty": ["l1", "l2", "elasticnet", None],
                    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
                },
                "Multi-layer Perceptron": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                    "activation": ["identity", "logistic", "tanh", "relu"],
                    "solver": ["lbfgs", "sgd", "adam"],
                    "alpha": [0.0001, 0.001, 0.01]
                },
                "XGBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 6, 9],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0]
                },
                "Explainable Boosting Classifier": {
                    "max_bins": [128, 256, 512],
                    "max_interaction_bins": [16, 32, 64],
                    "interactions": [5, 10, 15],
                    "learning_rate": [0.001, 0.01, 0.1]
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            }
            
            with st.spinner(f"Performing {tuning_method} for {model_to_tune}..."):
                if tuning_method == "Randomized Search":
                    from sklearn.model_selection import RandomizedSearchCV
                    search = RandomizedSearchCV(
                        st.session_state.models[model_to_tune],
                        param_distributions=param_grids[model_to_tune],
                        n_iter=10,
                        cv=5,
                        random_state=42,
                        n_jobs=-1
                    )
                elif tuning_method == "Grid Search":
                    from sklearn.model_selection import GridSearchCV
                    search = GridSearchCV(
                        st.session_state.models[model_to_tune],
                        param_grid=param_grids[model_to_tune],
                        cv=5,
                        n_jobs=-1
                    )
                elif tuning_method == "Bayesian Optimization":
                    from skopt import BayesSearchCV
                    search = BayesSearchCV(
                        st.session_state.models[model_to_tune],
                        search_spaces=param_grids[model_to_tune],
                        n_iter=10,
                        cv=5,
                        random_state=42,
                        n_jobs=-1
                    )
                
                search.fit(X_train, y_train)
                
                # Update the model with best parameters
                st.session_state.models[model_to_tune] = search.best_estimator_
                
                # Display results
                st.subheader("Tuning Results")
                st.write(f"Best parameters: {search.best_params_}")
                st.write(f"Best score (CV): {search.best_score_:.3f}")
                
                # Evaluate on test set
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                y_pred = search.best_estimator_.predict(X_test)
                
                from sklearn.metrics import classification_report
                st.write("Classification Report on Test Set:")
                st.text(classification_report(y_test, y_pred))
                
                st.success(f"{model_to_tune} tuned successfully!")
                
        except Exception as e:
            st.error(f"Error during hyperparameter tuning: {str(e)}")

# Step 8: Model Evaluation
elif current_step == steps[7] and 'models' in st.session_state and st.session_state.models:
    st.header("8. Model Evaluation")
    
    model_to_evaluate = st.selectbox("Select model to evaluate:", list(st.session_state.models.keys()))
    
    if st.button(f"Evaluate {model_to_evaluate}"):
        try:
            model = st.session_state.models[model_to_evaluate]
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
            # Classification report
            st.subheader("Classification Report")
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
            # ROC Curve (if probabilities available)
            if y_prob is not None:
                st.subheader("ROC Curve")
                fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='Yes')
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            
            # Precision-Recall Curve
            if y_prob is not None:
                st.subheader("Precision-Recall Curve")
                precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label='Yes')
                
                fig, ax = plt.subplots()
                ax.plot(recall, precision, color='blue', lw=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error during model evaluation: {str(e)}")

# Step 9: Explainability
elif current_step == steps[8] and 'models' in st.session_state and st.session_state.models:
    st.header("9. Model Explainability")
    
    model_to_explain = st.selectbox("Select model to explain:", list(st.session_state.models.keys()))
    
    explain_method = st.selectbox("Select explanation method:",
                                ["Feature Importance", 
                                 "LIME (Local Interpretable Model-agnostic Explanations)",
                                 "SHAP (SHapley Additive exPlanations)",
                                 "Decision Rules (for tree-based models)"])
    
    if st.button(f"Explain {model_to_explain}"):
        try:
            model = st.session_state.models[model_to_explain]
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            y_train = st.session_state.y_train
            
            if explain_method == "Feature Importance":
                st.subheader("Feature Importance")
                
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    st.write("Feature Importances:")
                    st.dataframe(importance_df)
                    
                    # Plot feature importances
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("This model doesn't support feature importances directly.")
            
            elif explain_method == "LIME (Local Interpretable Model-agnostic Explanations)":
                st.subheader("LIME Explanation")
                
                # Select an instance to explain
                instance_idx = st.slider("Select instance to explain:", 
                                       0, len(X_test)-1, 
                                       min(len(X_test)//2, 100))
                
                explainer = lime_tabular.LimeTabularExplainer(
                    X_train.values,
                    feature_names=X_train.columns,
                    class_names=np.unique(y_train),
                    mode='classification'
                )
                
                exp = explainer.explain_instance(
                    X_test.iloc[instance_idx].values,
                    model.predict_proba,
                    num_features=10
                )
                
                # Show explanation in Streamlit
                st.write(f"Explanation for instance {instance_idx}:")
                st.write(f"Actual class: {y_train.iloc[instance_idx]}")
                st.write(f"Predicted class: {model.predict([X_test.iloc[instance_idx]])[0]}")
                
                # Display explanation as HTML
                from IPython.display import HTML
                html = exp.as_html()
                st.components.v1.html(html, height=800, scrolling=True)
            
            elif explain_method == "SHAP (SHapley Additive exPlanations)":
                st.subheader("SHAP Explanation")
                
                import shap
                
                # Create SHAP explainer
                if hasattr(model, "predict_proba"):
                    explainer = shap.Explainer(model, X_train)
                    shap_values = explainer(X_test)
                    
                    # Summary plot
                    st.write("SHAP Summary Plot:")
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, X_test, plot_type="bar")
                    st.pyplot(fig)
                    
                    # Force plot for first instance
                    st.write("SHAP Force Plot for First Instance:")
                    shap.initjs()
                    force_plot = shap.force_plot(
                        explainer.expected_value[1],
                        shap_values[0,:,1].values,
                        X_test.iloc[0,:],
                        matplotlib=True
                    )
                    st.pyplot(force_plot)
                else:
                    st.warning("SHAP explainer requires predict_proba method.")
            
            elif explain_method == "Decision Rules (for tree-based models)":
                st.subheader("Decision Rules")
                
                if hasattr(model, "tree_") or hasattr(model, "estimators_"):
                    from sklearn.tree import export_text
                    
                    if hasattr(model, "estimators_"):  # Random Forest
                        tree_idx = st.slider("Select tree to visualize:", 
                                           0, len(model.estimators_)-1, 
                                           0)
                        tree_rules = export_text(model.estimators_[tree_idx], 
                                               feature_names=list(X_train.columns))
                    else:  # Single Decision Tree
                        tree_rules = export_text(model, 
                                               feature_names=list(X_train.columns))
                    
                    st.text_area("Decision Rules:", tree_rules, height=300)
                else:
                    st.warning("This model doesn't support decision rules visualization.")
            
        except Exception as e:
            st.error(f"Error during model explanation: {str(e)}")

# Final message if no data loaded
else:
    if current_step != steps[0]:
        st.warning("Please load data first from the 'Data Loading & Selection' step.")