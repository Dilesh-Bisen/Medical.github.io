from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import difflib

app = Flask(__name__)

# Load models and datasets
model = pickle.load(open("04_Medical/Output_file/model.pkl", "rb"))
symptom_dict = pickle.load(open("04_Medical/Output_file/symptom_dict.pkl", "rb"))
disease_dict = pickle.load(open("04_Medical/Output_file/disease_dict.pkl", "rb"))

prec = pd.read_csv('04_Medical/DataSets/precautions.csv')
work = pd.read_csv('04_Medical/DataSets/workout.csv')
descr = pd.read_csv('04_Medical/DataSets/description.csv')
medi = pd.read_csv('04_Medical/DataSets/medications.csv')
diet = pd.read_csv('04_Medical/DataSets/diets.csv')


def format_list(items):
    return '\n'.join(item.replace('[', '').replace(']', '').replace('\'', '') for item in items)


def predict_symptoms(symptom_input):
    ip_vector = np.zeros(len(symptom_dict))
    valid_symptoms = set(symptom_dict.keys())
    invalid_symptoms = []
    corrected_symptoms = []

    for symptom in symptom_input:
        if symptom in valid_symptoms:
            ip_vector[symptom_dict[symptom]] = 1
        else:
            matches = difflib.get_close_matches(symptom, valid_symptoms, n=1, cutoff=0.8)
            if matches:
                suggestion = matches[0]
                corrected_symptoms.append(suggestion)
            else:
                invalid_symptoms.append(symptom)

    if invalid_symptoms:
        print(f"Invalid symptoms: {', '.join(invalid_symptoms)}")

    if not ip_vector.any():
        return {"Error": "No valid symptoms provided."}

    try:
        prediction_index = model.predict([ip_vector])[0]
        disease_name = disease_dict[prediction_index]

        description = descr[descr['Disease'] == disease_name]['Description'].values[0]
        precautions = format_list(prec[prec['Disease'] == disease_name].iloc[:, 2:].values.flatten().tolist())
        medications = medi[medi['Disease'] == disease_name]['Medication'].values[0]
        if isinstance(medications, str):
            medications = format_list(medications.split(", "))
        workouts = work[work['disease'] == disease_name]['workout'].values[0]
        if isinstance(workouts, str):
            workouts = format_list(workouts.split(", "))
        diets = diet[diet['Disease'] == disease_name]['Diet'].values[0]
        if isinstance(diets, str):
            diets = format_list(diets.split(", "))

        return {
            'Disease': disease_name,
            'Description': description,
            'Precautions': precautions,
            'Medications': medications,
            'Workouts': workouts,
            'Diets': diets
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"Error": "An error occurred while processing the prediction or retrieving data."}


@app.route("/", methods=["GET"])
def index():
    return render_template("04_Medical/medical_recommendation/templates/index.html")


@app.route("/predict", methods=["POST"])
def predict():
    symptom_input = request.json.get('symptoms', [])
    result = predict_symptoms(symptom_input)
    return jsonify(result)


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    symptoms = list(symptom_dict.keys())
    return jsonify(symptoms)


if __name__ == "__main__":
    app.run(debug=True)
