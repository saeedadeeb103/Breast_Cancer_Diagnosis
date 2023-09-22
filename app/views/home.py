from flask import render_template, request, redirect, url_for
from app.models.models import Patient, Tumor, DiagnosisResults
from flask import Blueprint
from sklearn.datasets import load_breast_cancer
import pandas as pd


# Create a Flask Blueprint for views 
home_bp = Blueprint('home', __name__)

# load the breast cancer dataset and initialize model_accuracy 
Data = load_breast_cancer()
model_accuracy = {}

# Create DataFrame using Pandas
df = pd.DataFrame(Data.data, columns=Data.feature_names)
df['label'] = Data.target 

