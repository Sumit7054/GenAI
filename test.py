from pandasai import Agent
import pandas as pd
import os
from pandasai.llm import BambooLLM
from langchain_ollama import OllamaLLM
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List
import torch
import matplotlib.pyplot as plt
from pandasai.llm.local_llm import LocalLLM
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix    
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import sklearn


#llm = OllamaLLM(model="qwen2.5-coder:32b")
llm = LocalLLM(
    model="qwen2.5-coder:7b-instruct",
    api_base="http://localhost:11434/v1"
)
df = pd.read_csv("C:\\Users\\Sumit Kumar\\OneDrive\\Desktop\\Gen AI POC\\TestArea\\EmployeeAttrition.csv")
agent = Agent(df, config={"llm":llm, "enable_cache": False, "verbose": True})
response_stats = agent.chat("Give the data statistics")
print("Data Statistics:", response_stats)


response_models = agent.chat(
    "Train at least 5 classification models (random forest, logistic regression) "
    "with Attrition as the target column and show performance metrics (accuracy, f1-score)."
)
print("Model Performance:", response_models)