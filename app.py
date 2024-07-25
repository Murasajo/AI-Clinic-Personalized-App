import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load datasets
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Helper functions
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[model.predict([input_vector])[0]]

# Build streamli app
st.title("AI Clinic App")

st.image('image.jpeg', use_column_width=True)

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the App mode", ["Home", "About", "Contact"])


if app_mode == "Home":
    st.header("Symptom Prediction")
    symptoms = st.text_input("Enter your symptoms (e.g., itching, skin_rash, fatigue)")
    if st.button("Predict"):
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [sym for sym in user_symptoms if sym in symptoms_dict]
        if len(user_symptoms) > 0:
            result = get_predicted_value(user_symptoms)
            st.write(f"Predicted Disease: {result}")

            desc, pre, med, die, wrkout = helper(result)

            with st.expander("Description"):
                st.write(desc)
            with st.expander("Precautions"):
                st.write("\n".join(f"- {precaution}" for precaution in pre))
            with st.expander("Medications"):
                st.write(med)
            with st.expander("Diet"):
                st.write(die)
            with st.expander("Workout"):
                st.write(wrkout)

        else:
            st.warning("Please enter valid symptoms.")


elif app_mode == "About":
    st.markdown("""

        ---

        ### **Overview**
        The AI Clinic App is an advanced tool designed to assist users in identifying potential diseases based on a set of symptoms. Leveraging machine learning algorithms and a comprehensive dataset, this app provides accurate predictions and valuable health insights, aiming to support informed decision-making and prompt medical consultation.

        ### **Key Features**

        1. **Symptom-Based Prediction**: Users can input a list of symptoms, and the app utilizes a pre-trained models (SVC, RandomForestClassifier, GradientBoostingClassifier, KNeighborsClassifier, 'MultinomialNB) model to analyze the input and predict the most likely disease.

        2. **Detailed Information**:
            - **Description**: Provides a detailed explanation of the predicted disease, enhancing user understanding of the condition.
            - **Precautions**: Lists actionable precautions to mitigate the risk or manage the symptoms associated with the predicted disease.
            - **Medications**: Offers information on commonly recommended medications for the disease, though users should consult a healthcare professional before starting any medication.
            - **Diet**: Suggests dietary changes that can help manage or alleviate symptoms related to the disease.
            - **Workout**: Recommends exercises and physical activities that may benefit users with the specific condition.

        3. **Interactive Interface**:
            - **Expander Sections**: Each type of information (Description, Precautions, Medications, Diet, Workout) is accessible through expandable sections, ensuring a clean and organized user interface.

        ### **Usage Instructions**

        1. **Input Symptoms**: Enter symptoms in the provided text box, separating each symptom with a comma. Ensure that the symptoms are listed as recognized in the app's symptom dictionary.

        2. **Predict Disease**: Click the "Predict" button to submit the symptoms. The app processes the input, performs predictions using the different models, and displays the results.

        3. **Review Results**: After prediction, expand each section (Description, Precautions, Medications, Diet, Workout) to view detailed information about the predicted disease.

        ### **Technical Details**

        - **Machine Learning Model**: The app employs a variety of models trained on a dataset of symptoms and diseases to make predictions.
        - **Datasets**: Comprehensive datasets include information on symptoms, diseases, precautions, medications, diets, and workouts.
        - **Technology Stack**: Built using Streamlit for the user interface, and Python libraries such as NumPy and pandas for data manipulation.

        ### **Disclaimer**

        The Disease Prediction App is intended for informational purposes only and does not replace professional medical advice. Users should consult healthcare professionals for a proper diagnosis and treatment.

        ---


    """)
elif app_mode == "Contact":
    st.header("Contact Us")
    st.write("Email: dushimimanamurasajoseph@gmail.com")

