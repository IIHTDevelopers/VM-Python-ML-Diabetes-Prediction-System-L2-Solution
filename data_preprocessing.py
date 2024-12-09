import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_preprocess(file_path):
    """Load, clean, and preprocess the dataset."""
    try:
        data = pd.read_csv(file_path)

        # Validate required columns
        required_columns = {'diabetes', 'gender', 'smoking_history', 'age', 'bmi', 'HbA1c_level', 'blood_glucose_level'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Dataset missing required columns: {required_columns - set(data.columns)}")

        # Define features and target
        X = data.drop('diabetes', axis=1)
        y = data['diabetes']

        # Categorical and numerical columns
        categorical_cols = ['gender', 'smoking_history']
        numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

        # Pipelines for preprocessing
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Combined preprocessing
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ])

        # Apply preprocessing
        X_preprocessed = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_preprocessed, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, preprocessor
    except Exception as e:
        raise RuntimeError(f"Error during data preprocessing: {e}")
