from conversation import HealthConversation
from utils import extract_json
from predictor import DiseasePredictor
import json
from datetime import datetime


predictor = DiseasePredictor()


def ask_required(prompt_text, value_type=str):

    while True:

        value = input(prompt_text).strip()

        if not value:

            print("Please enter a value.")

            continue

        try:

            return value_type(value)

        except ValueError:

            print("Please enter a valid value.")


def infer_season(now):

    month = now.month

    if month in (3, 4, 5):

        return "Spring"

    if month in (6, 7, 8):

        return "Summer"

    if month in (9, 10):

        return "Post-Monsoon"

    return "Winter"


def normalize_optional_text(value, default="None"):

    cleaned = value.strip()

    return cleaned if cleaned else default


def ask_comorbidity():

    allowed_comorbidities = {
        "kidney disease": "Kidney Disease",
        "heart disease": "Heart Disease",
        "multiple": "Multiple",
        "obesity": "Obesity",
        "hypertension": "Hypertension",
        "diabetes": "Diabetes"
    }

    while True:

        value = input("Comorbidity (Kidney Disease, Heart Disease, Multiple, Obesity, Hypertension, Diabetes, or blank for nan): ").strip().lower()

        if not value:

            return "nan"

        if value in allowed_comorbidities:

            return allowed_comorbidities[value]

        print("Please enter Kidney Disease, Heart Disease, Multiple, Obesity, Hypertension, Diabetes, or leave blank for nan.")


def ask_gender():

    allowed_genders = {
        "female": "F",
        "other": "O",
        "male": "M"
    }

    while True:

        value = input("Gender (Female, Other, Male): ").strip().lower()

        if value in allowed_genders:

            return allowed_genders[value]

        print("Please enter Female, Other, or Male.")


def ask_smoking_status():

    allowed_smoking_status = {
        "never": "Never",
        "former": "Former",
        "current": "Current"
    }

    while True:

        value = input("Smoking status (Never, Former, Current): ").strip().lower()

        if value in allowed_smoking_status:

            return allowed_smoking_status[value]

        print("Please enter Never, Former, or Current.")


def ask_alcohol_use():

    allowed_alcohol_use = {
        "nan": "nan",
        "regular": "Regular",
        "occasional": "Occasional",
        "heavy": "Heavy"
    }

    while True:

        value = input("Alcohol use (nan, Regular, Occasional, Heavy): ").strip().lower()

        if not value:

            return "nan"

        if value in allowed_alcohol_use:

            return allowed_alcohol_use[value]

        print("Please enter nan, Regular, Occasional, or Heavy.")


def parse_height_to_meters(raw_height):

    height = float(raw_height)

    if height > 10:

        return height / 100.0

    return height


def calculate_bmi(height_meters, weight_kg):

    bmi = weight_kg / (height_meters ** 2)

    return round(bmi, 1)


def normalize_patient_symptoms(patient_data):

    model_symptoms = predictor.normalize_symptoms(patient_data.get("symptoms", ""))
    dictionary_symptoms = predictor.normalize_dictionary_symptoms(patient_data.get("symptoms", ""))

    normalized_symptoms = []

    for symptom in model_symptoms + dictionary_symptoms:
        if symptom not in normalized_symptoms:
            normalized_symptoms.append(symptom)

    patient_data["symptoms"] = ", ".join(normalized_symptoms) if normalized_symptoms else "Unknown"

    return patient_data


def add_treatment_suggestion(patient_data):

    suggestion = predictor.suggest_treatment(patient_data)

    patient_data["suggested_disease"] = suggestion["disease"]
    patient_data["matched_symptoms"] = suggestion["matched_symptoms"]
    patient_data["match_count"] = suggestion["match_count"]
    patient_data["match_ratio"] = suggestion["match_ratio"]
    patient_data["suggested_treatment"] = suggestion["treatment"]

    print("\nSymptom Match Result\n")
    print(f"Best Matching Disease: {suggestion['disease']}")
    print(f"Matched Symptoms: {', '.join(suggestion['matched_symptoms']) if suggestion['matched_symptoms'] else 'None'}")
    print(f"Match Count: {suggestion['match_count']}")
    print(f"Match Ratio: {suggestion['match_ratio']:.3f}")
    print(f"Suggested Treatment: {suggestion['treatment']}")

    return patient_data


def collect_patient_context():

    print("========== Patient Intake ==========")

    age = ask_required("Age: ", int)
    gender = ask_gender()
    comorbidity = ask_comorbidity()
    smoking_status = ask_smoking_status()
    alcohol_use = ask_alcohol_use()

    while True:

        try:

            height_m = parse_height_to_meters(ask_required("Height (cm or m): "))

            if height_m <= 0:

                raise ValueError

            break

        except ValueError:

            print("Please enter a valid height.")

    while True:

        try:

            weight_kg = float(ask_required("Weight (kg): "))

            if weight_kg <= 0:

                raise ValueError

            break

        except ValueError:

            print("Please enter a valid weight.")

    patient_context = {
        "age": age,
        "gender": gender,
        "season": infer_season(datetime.now()),
        "comorbidity": comorbidity,
        "smoking_status": smoking_status,
        "alcohol_use": alcohol_use,
        "bmi": calculate_bmi(height_m, weight_kg)
    }

    print("\nDemographic intake collected. Now collecting symptoms.\n")

    return patient_context


def finalize_symptoms(bot, patient_context):

    final_prompt = (
        "The symptom interview is complete. Return ONLY valid JSON for the final patient record. "
        "Include the symptoms that have been discussed. Do not ask any more questions. "
        "Use this schema: age, gender, season, comorbidity, smoking_status, alcohol_use, bmi, symptoms."
    )

    reply = bot.chat(final_prompt)
    patient_data = extract_json(reply)

    if patient_data is None:
        return None, reply

    patient_data = {**patient_data, **patient_context}
    patient_data = normalize_patient_symptoms(patient_data)

    return patient_data, reply


def main_cli():
    patient_context = collect_patient_context()
    bot = HealthConversation(patient_context=patient_context)

    user_input = "Begin the symptom interview. Ask only about the patient's current symptoms."
    turn_count = 0
    max_symptom_turns = 12

    print("========== AI Health Assistant ==========\n")
    print("Type 'done' when the symptom interview is complete.")

    while True:

        if user_input.strip().lower() in {"done", "finish", "stop", "no more"}:

            patient_data, final_reply = finalize_symptoms(bot, patient_context)

            if patient_data is not None:

                patient_data = normalize_patient_symptoms(patient_data)

                print("\nDoctor:")
                print(final_reply)

                print("\n========== Patient Data ==========")

                print("\nPatient Information\n")

                print(json.dumps(patient_data, indent=4))
                patient_data = add_treatment_suggestion(patient_data)
                with open("outputs/patient.json", "w") as f:
                    json.dump(patient_data, f, indent=4)

                break

            print("\nDoctor:")
            print(final_reply)

            print("\nThe model did not return JSON yet. Please type 'done' again or provide one more symptom detail.")

            user_input = input("\nYou: ")

            continue

        reply = bot.chat(user_input)

        
        patient_data = extract_json(reply)

        if patient_data is not None:

            patient_data = {**patient_data, **patient_context}
            patient_data = normalize_patient_symptoms(patient_data)

            print("\n========== Patient Data ==========\n")

            
            print("\nPatient Information\n")
            

            print(json.dumps(patient_data, indent=4))
            patient_data = add_treatment_suggestion(patient_data)
            with open("outputs/patient.json", "w") as f:
                json.dump(patient_data, f, indent=4)

            break

        print("\nDoctor:")
        print(reply)

        turn_count += 1

        if turn_count >= max_symptom_turns:

            print("\nSymptom interview limit reached. Finalizing now.")

            user_input = "done"

            continue

        user_input = input("\nYou: ")

    print("\nConversation Finished.")


if __name__ == "__main__":
    main_cli()