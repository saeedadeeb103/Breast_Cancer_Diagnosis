from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class DiagnosisService:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def diagnosis_tumor(self, patient, tumor):
        features = tumor.features
        scaled_features = self.scaler.transform([features])
        predict = self.model.predict(scaled_features)
        result = 'Malignant' if predict[0] == 0 else 'Benign'
        return result
    
