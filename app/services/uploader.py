from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np 
from io import StringIO
import csv
from sklearn.preprocessing import StandardScaler
from flask import request


class UploadHandler: 
    def __init__(self, models):
        self.scaler = StandardScaler()
        self.models = models

    def process_upload(self, file, model_name, cancer_data, x_trained, y_trained):
        self.x_train = x_trained
        self.x_train_scaled = self.scaler.fit_transform(self.x_train)
        self.y_trained = y_trained
        if file.filename != '':

            # Saving the File 
            file.save(file.filename)

            # Load the uploaded data and check the columns names 
            uploaded_data = pd.read_csv(file.filename)
            uploaded_columns = uploaded_data.columns.tolist()
            uploaded_columns = np.array(uploaded_columns, dtype='<U23')
            if not np.array_equal(uploaded_columns, cancer_data.feature_names):
                uploaded_columns = cancer_data.feature_names
            
            if set(uploaded_columns).issubset(set(cancer_data.feature_names)):
                # Keep only the columns which matchs the original names
                try:
                    uploaded_data = uploaded_data[cancer_data.feature_names]
                except:
                    return "Uploaded Data Containes Missing Columns"

                # Check if There is a Missing Values 
                if uploaded_data.isnull().values.any():
                    return "Uploaded Data Containes Missing Values"
                
                # Create a Buffer for the updated csv Data 
                output_buffer = StringIO()
                csv_writer = csv.writer(output_buffer)

                # Write the Head Row for the Updated CSV File 
                uploaded_header = list(uploaded_data.columns) + ['Result']
                csv_writer.writerow(uploaded_header)

                # Iterate thorugh each row in the uploaded data 
                for _, row in uploaded_data.iterrows():
                    features = row.values
                    scaled_feature = self.scaler.transform([features])
                    model = self.models[model_name]

                    # check if the model has been fitted, and fit it if not 

                    if not hasattr(model, 'classes_'):
                        model.fit(self.x_train_scaled, self.y_trained)
                    
                    prediction = model.predict(scaled_feature)
                    result = 'Malignant' if prediction[0] == 0 else 'Benign'
                    updated_row = list(features) + [result]
                    csv_writer.writerow(updated_row)

                output_buffer.seek(0)

                return output_buffer
            else:
                return "Uploaded Data Columns Do Not Match the Original features."
            