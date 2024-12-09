from data_preprocessing import load_and_preprocess
from model_training import train_and_evaluate
from prediction import predict_from_file
import joblib

def main():
    try:
        # Step 1: Preprocessing
        print("Loading and preprocessing the data...")
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess('diabetes_prediction_dataset.csv')

        # Save the preprocessor for later use
        joblib.dump(preprocessor, 'preprocessor.pkl')
        print("Preprocessing complete.\n")

        # Step 2: Train the model
        print("Training the model...")
        model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)

        # Save the trained model
        joblib.dump(model, 'diabetes_model.pkl')
        print("Model training complete. Model saved as 'diabetes_model.pkl'.\n")

        # Step 3: Make predictions for individuals from the input file
        print("Making predictions for individuals from the input file...")
        predictions = predict_from_file('persons_for_prediction.csv', 'preprocessor.pkl', 'diabetes_model.pkl')

        # Add accuracy explicitly to metrics dictionary
        metrics["accuracy"] = round(metrics.get("accuracy", 0), 2)

        # Display predictions
        print("\nPredictions for individuals:")
        for person, prediction in predictions.items():
            print(f"{person}: {prediction}")

        return metrics, predictions
    except Exception as e:
        print(f"An error occurred in the main workflow: {e}")
        return {}, {}

if __name__ == "__main__":
    main()
