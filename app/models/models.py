from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ModelAccuracy:
    def __init__(self, models, scaler):
        self.models = models 
        self.scaler = scaler
        self.model_accuracy = {}
        self.setup_accuracy()

    def setup_accuracy(self, X, Y):
        for model_name, model in self.models.items():
            # Check if the model is not fitted 
            if not hasattr(model, 'classes_'):
                if model_name == 'Logistch Regression':
                    x_train, x_test, y_train, y_test = train_test_split(X, Y, )