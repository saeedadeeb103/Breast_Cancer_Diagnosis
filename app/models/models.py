from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ModelAccuracy:
    """
    A class for calculating and storing accuracy scores of machine learning models.

    Attributes:
        models (dict): A dictionary of machine learning models.
        scaler (StandardScaler): A StandardScaler object used for feature scaling.
        model_accuracy (dict): A dictionary to store accuracy scores of each model.

    Methods:
        setup_accuracy(X, Y):
            Calculate the accuracy scores of machine learning models on a given dataset.

    """

    def __init__(self, models, scaler):
        """
        Initializes the ModelAccuracy class with machine learning models and a scaler.

        Args:
            models (dict): A dictionary where keys are model names, and values are machine learning models.
            scaler (StandardScaler): A StandardScaler object used for feature scaling.
        """
        self.models = models 
        self.scaler = scaler
        self.model_accuracy = {}

    def setup_accuracy(self, X, Y):
        """
        Calculate the accuracy scores of machine learning models on a given dataset.

        Args:
            X (array-like): The feature matrix.
            Y (array-like): The target labels.

        Returns:
            dict: A dictionary containing model names as keys and their accuracy scores as values.
        """
        for model_name, model in self.models.items():
            # Check if the model is not fitted 
            if not hasattr(model, 'classes_'):
                if model_name == 'Logistic Regression':
                    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
                elif model_name == 'Random Forest':
                    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                elif model_name == 'SVM':
                    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                else:
                    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

                x_train_scaled = self.scaler.fit_transform(x_train)
                x_test_scaled = self.scaler.transform(x_test)

                model.fit(x_train_scaled, y_train)
                y_pred = model.predict(x_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                self.model_accuracy[model_name] = accuracy

        return self.model_accuracy
