import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import torch


def safe_log(x):
    return np.log1p(np.clip(x, 0, None))


def load_data():
    file_path = "candidateschallenge/candidates_data.csv"
    image_folder = "candidateschallenge/spacecraft_images"
    df = pd.read_csv(file_path)

    # Separate features and target
    X = df.drop(['target', 'description'], axis=1)
    y = df['target']

    # Log transform the target
    y = safe_log(y)

    # Identify numeric and categorical columns
    categorical_features = X.select_dtypes(
        include=['object', 'category']).columns.tolist()
    categorical_features.append('source_id')
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove('source_id')

    # Remove columns with all missing values from numeric_features
    X_numeric = X[numeric_features]
    all_missing_cols = X_numeric.columns[X_numeric.isna().all()].tolist()
    valid_numeric_features = [
        col for col in numeric_features if col not in all_missing_cols]

    # Print information about removed columns
    if all_missing_cols:
        print(f"Removing columns with all missing values: {all_missing_cols}")

    # Create preprocessing steps with constant fill for numeric features
    numeric_transformer = Pipeline(steps=[
        # Changed to constant strategy
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            # Use valid_numeric_features
            ('num', numeric_transformer, valid_numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
    ])

    X_train, X_test, y_train, y_test, desc_train, desc_test = train_test_split(
        X, y, df['description'], test_size=0.2, random_state=42)

    # Fit and transform the data
    X_train_preprocessed = full_pipeline.fit_transform(X_train)
    X_test_preprocessed = full_pipeline.transform(X_test)
    joblib.dump(full_pipeline, 'preprocessing_pipeline.joblib')
    print('created')

    return X_train_preprocessed, y_train, X_test_preprocessed, y_test, desc_train, desc_test, image_folder


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2


def cross_validate(model, X, y, cv=5):
    scores = cross_val_score(
        model, X, y, cv=cv, scoring='neg_mean_squared_error')
    return -scores.mean()


def custom_collate(batch):
    tabular, text_inputs, images, targets = zip(*batch)

    tabular = torch.stack(tabular)
    targets = torch.stack(targets)
    images = torch.stack(images)

    max_length = max(text_input['input_ids'].size(0)
                     for text_input in text_inputs)

    padded_input_ids = []
    padded_attention_mask = []

    for text_input in text_inputs:
        input_ids = text_input['input_ids']
        attention_mask = text_input['attention_mask']

        pad_length = max_length - input_ids.size(0)

        if pad_length > 0:
            padded_input_ids.append(
                torch.cat([input_ids, torch.zeros(
                    pad_length, dtype=torch.long)])
            )
            padded_attention_mask.append(
                torch.cat([attention_mask, torch.zeros(
                    pad_length, dtype=torch.long)])
            )
        else:
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)

    combined_text_inputs = {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask)
    }

    return tabular, combined_text_inputs, images, targets
