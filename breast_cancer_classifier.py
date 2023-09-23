from flask import Flask, render_template, request
import csv
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from app.services.uploader import UploadHandler
from app.models.models import ModelAccuracy

app = Flask(__name__)

class BreastCancerApp:
    def __init__(self, app):
        self.app = app
        self.setup_routes()
        self.setup_models()

    def setup_routes(self):
        self.app.add_url_rule('/', view_func=self.index)
        self.app.add_url_rule('/predict', view_func=self.predict, methods=['POST'])
        self.app.add_url_rule('/upload', view_func=self.upload, methods=['GET', 'POST'])
        self.app.add_url_rule('/upload_result', view_func=self.upload_result)
    
    def setup_models(self):
        self.model_accuracies = {}
        self.breast_cancer_data = load_breast_cancer()
        # Create a DataFrame
        self.df = pd.DataFrame(self.breast_cancer_data.data, columns=self.breast_cancer_data.feature_names)
        self.df['label'] = self.breast_cancer_data.target
        # Split the data
        X = self.df.drop(columns='label', axis=1)
        Y = self.df['label']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

        # Standardize features
        self.scaler = StandardScaler()
        self.x_train_scaled = self.scaler.fit_transform(self.x_train)
        self.x_test_scaled = self.scaler.transform(self.x_test)

        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(C=1, penalty='l1', solver='saga', multi_class='ovr', max_iter=150, random_state=40),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=2),
            'SVM': SVC(random_state=2),
            'XGBoost': XGBClassifier(random_state=2),
        }
        
        model_accuracy = ModelAccuracy(models=self.models,
                                       scaler= self.scaler)
        self.model_accuracies = model_accuracy.setup_accuracy(X, Y)

    def index(self):
        feature_names = self.df.columns[:-1]
        return render_template('index.html', feature_names=feature_names, model_accuracies=self.model_accuracies)

    def predict(self):
        if request.method == 'POST':
            model_name = request.form['model']
            if model_name == 'Random Forest':
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
                self.x_train_scaled = self.scaler.fit_transform(self.x_train)
            features = [float(request.form[feature]) for feature in self.df.columns[:-1]]
            scaled_features = self.scaler.transform([features])
            model = self.models[model_name]

            # Fit the model on the training data (if not already fitted)
            if not hasattr(model, 'classes_'):
                model.fit(self.x_train_scaled, self.y_train)

            prediction = model.predict(scaled_features)
            result = 'Malignant' if prediction[0] == 0 else 'Benign'
            return render_template('result.html', model=model_name, result=result)

    def upload(self):
        self.breast_cancer_data = load_breast_cancer()
        model_name = ""  # Default model_name
        if request.method == 'POST':
            uploaded_file = request.files['file']
            model_name = request.form['model']
            data_list = []
            if uploaded_file.filename != '':
                # Save the uploaded file
                Upload_handler = UploadHandler(self.models)
                output_buffer = Upload_handler.process_upload(file=uploaded_file, model_name=model_name, cancer_data=self.breast_cancer_data, x_trained=self.x_train, y_trained=self.y_train)

                if isinstance(output_buffer, str):
                    return output_buffer
                
                # Render the 'upload_result.html' template with the updated CSV data
                return render_template('upload_result.html', data=csv.DictReader(output_buffer), model=model_name)
        return render_template('upload.html')

    def upload_result(self):
        # This route can be implemented if needed for displaying upload results
        pass

if __name__ == '__main__':
    breast_cancer_app = BreastCancerApp(app)  # Create an instance of the class
    app.run(debug=True)  # Run the Flask app
