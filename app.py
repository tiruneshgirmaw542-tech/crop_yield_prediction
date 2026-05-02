from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = None
preprocessor = None
model_error = None

try:
    with open('crop_model.pkl', 'rb') as f:
        model = pickle.load(f)
except ModuleNotFoundError as exc:
    model_error = f"Model could not be loaded because a dependency is missing: {exc.name}."
except Exception as exc:
    model_error = f"Model could not be loaded: {exc}"

try:
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except FileNotFoundError:
    model_error = model_error or 'Preprocessor file preprocessor.pkl is missing.'
except ModuleNotFoundError as exc:
    model_error = f"Preprocessor could not be loaded because a dependency is missing: {exc.name}."
except Exception as exc:
    model_error = model_error or f"Preprocessor could not be loaded: {exc}"

FEATURE_FIELDS = [
    ('Area', 'Area (country)'),
    ('Item', 'Item (crop/product)'),
    ('Year', 'Year'),
    ('average_rain_fall_mm_per_year', 'Average rainfall (mm/year)'),
    ('pesticides_tonnes', 'Pesticides used (tonnes)'),
    ('avg_temp', 'Average temperature')
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = model_error
    values = {field: '' for field, _ in FEATURE_FIELDS}

    if request.method == 'POST':
        try:
            # Collect and validate inputs
            values = {field: request.form.get(field, '').strip() for field, _ in FEATURE_FIELDS}
            
            # Create a dictionary for the DataFrame, ensuring numeric types are converted
            data = {
                'Area': [values['Area']],
                'Item': [values['Item']],
                'Year': [float(values['Year'])],
                'average_rain_fall_mm_per_year': [float(values['average_rain_fall_mm_per_year'])],
                'pesticides_tonnes': [float(values['pesticides_tonnes'])],
                'avg_temp': [float(values['avg_temp'])],
            }

            if model is None or preprocessor is None:
                raise Exception("Prediction model or preprocessor is unavailable.")

            df_sample = pd.DataFrame(data)
            transformed = preprocessor.transform(df_sample)
            prediction = model.predict(transformed)[0]

        except ValueError as exc:
            error = f'Invalid numeric input: Please ensure Year, Rainfall, Pesticides, and Temperature are numbers. Details: {exc}'
        except Exception as exc:
            error = f'Prediction error: {exc}'

    return render_template('index.html', prediction=prediction, error=error, values=values, feature_fields=FEATURE_FIELDS)

import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)