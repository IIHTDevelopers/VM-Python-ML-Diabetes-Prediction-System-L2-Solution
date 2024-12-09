from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train and evaluate the model."""
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_test)
        metrics = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)

        print("Model Evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy:.2f}")

        return model, metrics
    except Exception as e:
        raise RuntimeError(f"Error during model training and evaluation: {e}")
