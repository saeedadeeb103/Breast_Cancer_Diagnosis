from flask import Flask, render_template, request, redirect, url_for
import csv
from io import StringIO
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load the breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Create a DataFrame
df = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
df['label'] = breast_cancer_data.target

# Split the data
X = df.drop(columns='label', axis=1)
Y = df['label']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=2),
    'Gradient Boosting': GradientBoostingClassifier(random_state=2),
    'SVM': SVC(random_state=2),
    'XGBoost': XGBClassifier(random_state=2),
}

@app.route('/')
def index():
    feature_names = breast_cancer_data.feature_names
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        model_name = request.form['model']
        features = [float(request.form[feature]) for feature in breast_cancer_data.feature_names]
        scaled_features = scaler.transform([features])
        model = models[model_name]
        
        # Fit the model on the training data (if not already fitted)
        if not hasattr(model, 'classes_'):
            model.fit(x_train_scaled, y_train)
        
        prediction = model.predict(scaled_features)
        result = 'Malignant' if prediction[0] == 0 else 'Benign'
        return render_template('result.html', model=model_name, result=result)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    model_name = ""  # Default model_name
    if request.method == 'POST':
        uploaded_file = request.files['file']
        data_list = []
        if uploaded_file.filename != '':
            # Save the uploaded file
            uploaded_file.save(uploaded_file.filename)

            # Load the uploaded data and check column names
            uploaded_data = pd.read_csv(uploaded_file.filename)
            uploaded_columns = uploaded_data.columns.tolist()
            uploaded_columns = np.array(uploaded_columns, dtype='<U23')
            if not np.array_equal(uploaded_columns, breast_cancer_data.feature_names):
                uploaded_columns = breast_cancer_data.feature_names
            # Check if the uploaded columns match the original features
            if set(uploaded_columns).issubset(set(breast_cancer_data.feature_names)):
                # Keep only the columns that match the original features
                try:
                    uploaded_data = uploaded_data[breast_cancer_data.feature_names]
                except: 
                    return "Uploaded data contains missing Column" 

                if uploaded_data.isnull().values.any():
                    return "Uploaded data contains missing values."

                # Create a buffer for the updated CSV data
                output_buffer = StringIO()
                csv_writer = csv.writer(output_buffer)

                # Write the header row for the updated CSV
                updated_header = list(uploaded_data.columns) + ['Result']
                csv_writer.writerow(updated_header)

                # Iterate through each row in the uploaded data
                for _, row in uploaded_data.iterrows():
                    features = row.values
                    scaled_features = scaler.transform([features])
                    model_name = request.form['model']  # Update model_name here
                    model = models[model_name]

                    # Check if the model has been fitted, and fit it if not
                    if not hasattr(model, 'classes_'):
                        model.fit(x_train_scaled, y_train)

                    prediction = model.predict(scaled_features)
                    result = 'Malignant' if prediction[0] == 0 else 'Benign'
                    updated_row = list(features) + [result]
                    csv_writer.writerow(updated_row)

                # Reset the buffer position
                output_buffer.seek(0)

                # Render the 'upload_result.html' template with the updated CSV data
                return render_template('upload_result.html', data=csv.DictReader(output_buffer), model=model_name)
            else:
                return "Uploaded data columns do not match the original features."

    return render_template('upload.html')




if __name__ == '__main__':
    app.run(debug=True)
