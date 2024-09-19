import streamlit as st
import pandas as pd
import numpy as np
import pickle
import difflib

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


# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f5;
    }
    .title {
        color: #4CAF50;
        font-size: 2em;
        text-align: center;
        margin-bottom: 20px;
    }
    .recommendation {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.markdown("<h1 class='title'>Medical Recommendation System</h1>", unsafe_allow_html=True)

# User input for symptoms
symptom_input = st.multiselect("Select Symptoms", options=list(symptom_dict.keys()))

if st.button("Get Recommendations"):
    result = predict_symptoms(symptom_input)
    if "Error" in result:
        st.error(result["Error"])
    else:
        st.subheader("Recommendations")
        st.markdown("<div class='recommendation'>", unsafe_allow_html=True)
        st.write(f"**Disease:** {result['Disease']}")
        st.write(f"**Description:** {result['Description']}")
        st.write(f"**Precautions:** {result['Precautions']}")
        st.write(f"**Medications:** {result['Medications']}")
        st.write(f"**Workouts:** {result['Workouts']}")
        st.write(f"**Diets:** {result['Diets']}")
        st.markdown("</div>", unsafe_allow_html=True)

# Display available symptoms
if st.checkbox("Show All Symptoms"):
    st.subheader("Available Symptoms")
    st.write(list(symptom_dict.keys()))
