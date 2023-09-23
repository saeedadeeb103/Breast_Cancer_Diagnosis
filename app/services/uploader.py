from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np 
from io import StringIO
import csv
from sklearn.preprocessing import StandardScaler
from flask import request

class UploadHandler:
    """
    Handles the processing of uploaded data files, applying a machine learning model for predictions.

    Attributes:
        scaler (StandardScaler): A StandardScaler object used for feature scaling.
        models (dict): A dictionary of machine learning models.

    Methods:
        process_upload(file, model_name, cancer_data, x_trained, y_trained):
            Process an uploaded data file, make predictions using a specified machine learning model,
            and return the results.

    """

    def __init__(self, models):
        """
        Initializes the UploadHandler with a StandardScaler and a dictionary of machine learning models.

        Args:
            models (dict): A dictionary where keys are model names, and values are the corresponding machine learning models.
        """
        self.scaler = StandardScaler()
        self.models = models

    def process_upload(self, file, model_name, cancer_data, x_trained, y_trained):
        """
        Process an uploaded data file, make predictions using a specified machine learning model,
        and return the results.

        Args:
            file (FileStorage): The uploaded file containing data for prediction.
            model_name (str): The name of the machine learning model to use for prediction.
            cancer_data: An object containing cancer data and feature names.
            x_trained (array-like): The training data features.
            y_trained (array-like): The training data labels.

        Returns:
            Union[str, StringIO]: A StringIO object containing the updated CSV data with predictions,
            or an error message as a string if there are issues with the uploaded data.

        """
        self.x_train = x_trained
        self.x_train_scaled = self.scaler.fit_transform(self.x_train)
        self.y_trained = y_trained
        if file.filename != '':

            # Saving the File 
            file.save(file.filename)

            # Load the uploaded data and check the column names 
            uploaded_data = pd.read_csv(file.filename)
            uploaded_columns = uploaded_data.columns.tolist()
            uploaded_columns = np.array(uploaded_columns, dtype='<U23')
            if not np.array_equal(uploaded_columns, cancer_data.feature_names):
                uploaded_columns = cancer_data.feature_names
            
            if set(uploaded_columns).issubset(set(cancer_data.feature_names)):
                # Keep only the columns which match the original names
                try:
                    uploaded_data = uploaded_data[cancer_data.feature_names]
                except:
                    return "Uploaded Data Contains Missing Columns"

                # Check if There are Missing Values 
                if uploaded_data.isnull().values.any():
                    return "Uploaded Data Contains Missing Values"
                
                # Create a Buffer for the updated CSV Data 
                output_buffer = StringIO()
                csv_writer = csv.writer(output_buffer)

                # Write the Header Row for the Updated CSV File 
                uploaded_header = list(uploaded_data.columns) + ['Result']
                csv_writer.writerow(uploaded_header)

                # Iterate through each row in the uploaded data 
                for _, row in uploaded_data.iterrows():
                    features = row.values
                    scaled_feature = self.scaler.transform([features])
                    model = self.models[model_name]

                    # Check if the model has been fitted, and fit it if not 
                    if not hasattr(model, 'classes_'):
                        model.fit(self.x_train_scaled, self.y_trained)
                    
                    prediction = model.predict(scaled_feature)
                    result = 'Malignant' if prediction[0] == 0 else 'Benign'
                    updated_row = list(features) + [result]
                    csv_writer.writerow(updated_row)

                output_buffer.seek(0)

                return output_buffer
            else:
                return "Uploaded Data Columns Do Not Match the Original Features."
