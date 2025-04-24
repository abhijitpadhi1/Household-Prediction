from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import joblib

# Load the trained model
MODEL_PATH = r"models\final_model.pkl"
loaded_model = joblib.load(MODEL_PATH)

# Fixed category list for OneHotEncoder (to match training time)
ocean_categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

# Column indices used in custom transformer
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

# Custom transformer for engineered features
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def preprocess_data(data):
    features = [
        "longitude", "latitude", "housing_median_age",
        "total_rooms", "total_bedrooms", "population",
        "households", "median_income"
    ]

    # Create dataframe with consistent column names
    housing = pd.DataFrame(data, columns=features + ["ocean_proximity"])

    # Define the numeric pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ])

    # Full pipeline including fixed categories for encoder
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, features),
        ("cat", OneHotEncoder(categories=[ocean_categories], handle_unknown="ignore"), ["ocean_proximity"]),
    ])

    # Fit and transform only once per request (no data leakage here as this is for inference only)
    housing_prepared = full_pipeline.fit_transform(housing)
    return housing_prepared

def predict_price(data):
    housing_prepared = preprocess_data(data)
    prediction = loaded_model.predict(housing_prepared)
    return prediction[0]
