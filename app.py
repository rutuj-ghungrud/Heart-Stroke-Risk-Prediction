from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
NUMERICAL_COLUMNS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL_COLUMNS = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
FORM_FIELDS = {
    "Age": ("age", int, 1, 120),
    "RestingBP": ("resting_bp", int, 50, 250),
    "Cholesterol": ("cholesterol", int, 50, 700),
    "FastingBS": ("fasting_bs", int, 0, 1),
    "MaxHR": ("max_hr", int, 40, 250),
    "Oldpeak": ("oldpeak", int, -3, 10),
}

app = Flask(__name__)

model = joblib.load(BASE_DIR / "KNN_heart.pkl")
prediction_scaler = joblib.load(BASE_DIR / "scaler.pkl")
feature_columns = joblib.load(BASE_DIR / "columns.pkl")

training_data = pd.read_csv(DATA_PATH)
if training_data.empty:
    raise ValueError(f"Training data is empty: {DATA_PATH}")

cholesterol_mean = training_data.loc[
    training_data["Cholesterol"] != 0, "Cholesterol"
].mean()
training_data["Cholesterol"] = (
    training_data["Cholesterol"].replace(0, cholesterol_mean).round(2)
)
training_data["RestingBP"] = (
    training_data["RestingBP"].replace(0, cholesterol_mean).round(2)
)

category_options = {
    column: sorted(training_data[column].dropna().unique().tolist())
    for column in CATEGORICAL_COLUMNS
}

encoded_training_data = pd.get_dummies(training_data, drop_first=True).astype(int)
initial_scaler = StandardScaler()
encoded_training_data[NUMERICAL_COLUMNS] = initial_scaler.fit_transform(
    encoded_training_data[NUMERICAL_COLUMNS]
)

training_feature_columns = [
    column for column in encoded_training_data.columns if column != "HeartDisease"
]
if training_feature_columns != feature_columns:
    raise ValueError(
        "The training data features do not match columns.pkl. "
        "Check that the model and data files belong to the same training run."
    )
if prediction_scaler.n_features_in_ != len(feature_columns):
    raise ValueError("scaler.pkl does not match the feature count in columns.pkl.")


def encode_input(values):
    row = pd.DataFrame([values])
    for column, options in category_options.items():
        row[column] = pd.Categorical(row[column], categories=options)

    encoded = pd.get_dummies(
        row, columns=CATEGORICAL_COLUMNS, drop_first=True
    ).astype(int)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    encoded[NUMERICAL_COLUMNS] = initial_scaler.transform(
        encoded[NUMERICAL_COLUMNS]
    )
    return encoded


def render_page(*, submitted=None, result=None, errors=None, status=200):
    return (
        render_template(
            "index.html",
            categories=category_options,
            submitted=submitted or {},
            result=result,
            errors=errors or [],
        ),
        status,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_page()

    submitted = request.form.to_dict()
    errors = []
    values = {}

    for column, (field_name, value_type, minimum, maximum) in FORM_FIELDS.items():
        raw_value = submitted.get(field_name, "").strip()
        try:
            value = value_type(raw_value)
        except ValueError:
            errors.append(f"Enter a valid value for {column}.")
            continue

        if not minimum <= value <= maximum:
            errors.append(f"{column} must be between {minimum} and {maximum}.")
            continue
        values[column] = value

    for column in CATEGORICAL_COLUMNS:
        field_name = column.lower()
        value = submitted.get(field_name, "")
        if value not in category_options[column]:
            errors.append(f"Choose a valid option for {column}.")
        else:
            values[column] = value

    if errors:
        return render_page(submitted=submitted, errors=errors, status=400)

    encoded = encode_input(values)
    prediction = int(model.predict(prediction_scaler.transform(encoded))[0])
    result = {
        "has_condition": prediction == 1,
        "label": (
            "Higher likelihood of heart disease"
            if prediction == 1
            else "Lower likelihood of heart disease"
        ),
    }
    return render_page(submitted=submitted, result=result)


if __name__ == "__main__":
    app.run()
