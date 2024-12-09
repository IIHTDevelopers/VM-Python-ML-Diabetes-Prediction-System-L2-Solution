import pandas as pd
import joblib

def predict_from_file(file_path, preprocessor_path, model_path):
    """Predict diabetes for individuals in a given CSV file."""
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        data = pd.read_csv(file_path)
        if 'name' not in data.columns:
            raise ValueError("Input data must contain a 'name' column.")

        names = data['name']
        X = data.drop('name', axis=1)

        X_preprocessed = preprocessor.transform(X)
        predictions = model.predict(X_preprocessed)

        results = {name: 'Diabetic' if pred == 1 else 'Non-Diabetic' for name, pred in zip(names, predictions)}
        return results
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")
