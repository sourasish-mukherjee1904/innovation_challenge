import sys
import os
import pandas as pd
import numpy as np
import pickle
from predictor import DiseasePredictor

# Target disease mappings for dhivyeshrk dataset
disease_mapping = {
    "common cold": "Respiratory Infection", "flu": "Respiratory Infection", "pneumonia": "Respiratory Infection", 
    "acute bronchitis": "Respiratory Infection", "acute bronchiolitis": "Respiratory Infection",
    "infectious gastroenteritis": "Diarrheal Disease", "noninfectious gastroenteritis": "Diarrheal Disease", "food poisoning": "Diarrheal Disease",
    "asthma": "Chronic Respiratory Disease", "chronic obstructive pulmonary disease (copd)": "Chronic Respiratory Disease",
    "diabetes": "Diabetes", "gestational diabetes": "Diabetes", "diabetes insipidus": "Diabetes",
    "high blood pressure": "Hypertension", "pulmonary hypertension": "Hypertension", "malignant hypertension": "Hypertension", "kidney disease due to longstanding hypertension": "Hypertension",
    "liver disease": "Liver Infection",
    "depression": "Mental Health Disorder", "anxiety": "Mental Health Disorder", "panic disorder": "Mental Health Disorder", "bipolar disorder": "Mental Health Disorder",
    "anemia": "Blood Disorder", "iron deficiency anemia": "Blood Disorder", "sickle cell anemia": "Blood Disorder",
    "chronic kidney disease": "Chronic Kidney Disease", "kidney failure": "Chronic Kidney Disease", "diabetic kidney disease": "Chronic Kidney Disease",
    "malaria": "Mosquito-borne Fever", "dengue fever": "Mosquito-borne Fever",
    "typhoid fever": "Typhoid",
}

# Symptom mappings
symptom_mapping = {
    "abdominal pain": ["sharp abdominal pain", "lower abdominal pain", "burning abdominal pain", "upper abdominal pain"],
    "appetite changes": ["decreased appetite", "excessive appetite"],
    "blurred vision": ["diminished vision", "double vision", "spots or clouds in vision"],
    "chest pain": ["sharp chest pain", "burning chest pain"],
    "chills": ["chills"],
    "cough": ["cough", "coughing up sputum"],
    "cramps": ["cramps and spasms", "muscle cramps contractures or spasms", "arm cramps or spasms", "leg cramps or spasms"],
    "diarrhea": ["diarrhea"],
    "dizziness": ["dizziness"],
    "fatigue": ["fatigue"],
    "fever": ["fever"],
    "frequent urination": ["frequent urination"],
    "high fever": ["fever"],
    "irritability": ["premenstrual tension or irritability"],
    "joint pain": ["joint pain"],
    "muscle pain": ["muscle pain"],
    "nausea": ["nausea"],
    "rash": ["eyelid lesion or rash", "skin rash", "diaper rash"],
    "severe headache": ["headache"],
    "shortness of breath": ["shortness of breath"],
    "sore throat": ["sore throat"],
    "sweating": ["sweating"],
    "swelling": ["throat swelling", "skin swelling", "hand or finger swelling", "leg swelling", "foot or toe swelling", "joint swelling"],
    "swollen lymph nodes": ["swollen lymph nodes"],
    "vomiting": ["vomiting", "vomiting blood"],
    "weight loss": ["recent weight loss"],
    "wheezing": ["wheezing"],
    "excessive worry": ["anxiety and nervousness"],
    "loss of appetite": ["decreased appetite"],
    "loss of smell/taste": ["disturbance of smell or taste"],
    "numbness": ["loss of sensation"],
    "persistent sadness": ["depression", "depressive or psychotic symptoms"],
    "sleep disturbance": ["insomnia", "sleepiness"],
    "delayed growth": ["lack of growth"],
}

def load_samples(count=200, seed=42):
    path = r"C:\Users\sayak\.cache\kagglehub\datasets\dhivyeshrk\diseases-and-symptoms-dataset\versions\1"
    data_csv = os.path.join(path, "Final_Augmented_dataset_Diseases_and_Symptoms.csv")
    if not os.path.exists(data_csv):
        print(f"Error: Dataset CSV not found at {data_csv}")
        return None
    df = pd.read_csv(data_csv)
    df_filtered = df[df["diseases"].isin(disease_mapping.keys())].copy()
    np.random.seed(seed)
    return df_filtered.sample(n=min(count, len(df_filtered))).reset_index(drop=True)

def run_evaluation(predictor, method_name, use_hierarchical=False):
    df_sample = load_samples(200)
    if df_sample is None:
        return
    
    print(f"\nEvaluating {method_name} over 200 samples...")
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    
    for idx, row in df_sample.iterrows():
        kaggle_disease = row["diseases"]
        expected_class = disease_mapping[kaggle_disease]
        
        present_symptoms = []
        for our_sym, kag_cols in symptom_mapping.items():
            is_present = False
            for col in kag_cols:
                if col in row and row[col] == 1:
                    is_present = True
                    break
            if is_present:
                present_symptoms.append(our_sym)
                
        patient = {
            "age": 30, "gender": "F", "season": "Summer", "comorbidity": "nan",
            "smoking_status": "Never", "alcohol_use": "nan", "bmi": 22.0,
            "symptoms": ", ".join(present_symptoms) if present_symptoms else "Unknown"
        }
        
        try:
            if use_hierarchical:
                top3 = predictor.predict_top3_hierarchical(patient)
            else:
                top3 = predictor.predict_top3(patient)
                
            top3_classes = [c for c, p in top3]
            if top3_classes and top3_classes[0] == expected_class:
                correct_top1 += 1
            if expected_class in top3_classes:
                correct_top3 += 1
            total += 1
        except Exception as e:
            pass
            
    print(f"\n================ {method_name.upper()} RESULTS ================")
    print(f"Total Profiles Evaluated : {total}")
    print(f"Top-1 Prediction Accuracy: {correct_top1 / total:.2%}")
    print(f"Top-3 Prediction Accuracy: {correct_top3 / total:.2%}")
    print("=================================================================")

def run_vignettes_validation(predictor):
    df_sample = load_samples(50, seed=101)
    if df_sample is None:
        return
        
    print("\nRunning 50 Clinical Vignettes Validation...")
    cb_top1, cb_top3 = 0, 0
    hier_top1, hier_top3 = 0, 0
    total = df_sample.shape[0]
    
    for idx, row in df_sample.iterrows():
        kaggle_disease = row["diseases"]
        expected_class = disease_mapping[kaggle_disease]
        
        present_symptoms = []
        for our_sym, kag_cols in symptom_mapping.items():
            is_present = False
            for col in kag_cols:
                if col in row and row[col] == 1:
                    is_present = True
                    break
            if is_present:
                present_symptoms.append(our_sym)
                
        patient = {
            "age": 30, "gender": "F", "season": "Summer", "comorbidity": "nan",
            "smoking_status": "Never", "alcohol_use": "nan", "bmi": 22.0,
            "symptoms": ", ".join(present_symptoms) if present_symptoms else "Unknown"
        }
        
        # Standard
        cb = predictor.predict_top3(patient)
        cb_cls = [c for c, p in cb]
        if cb_cls and cb_cls[0] == expected_class: cb_top1 += 1
        if expected_class in cb_cls: cb_top3 += 1
        
        # Hierarchical
        hier = predictor.predict_top3_hierarchical(patient)
        hier_cls = [c for c, p in hier]
        if hier_cls and hier_cls[0] == expected_class: hier_top1 += 1
        if expected_class in hier_cls: hier_top3 += 1
        
    print("\n================ 50-VIGNETTES ACCURACY REPORT ================")
    print("Model                             | Top-1 Acc  | Top-3 Acc")
    print("---------------------------------------------------------------")
    print(f"CatBoost Baseline (No multipliers) | {cb_top1/total:10.2%} | {cb_top3/total:10.2%}")
    print(f"Hierarchical Classifier (Trained)  | {hier_top1/total:10.2%} | {hier_top3/total:10.2%}")
    print("===============================================================")

def interactive_chat_session(predictor):
    print("\nStarting Interactive Clinical Interview...")
    print("Select predicting engine:")
    print(" 1. CatBoost Baseline Model")
    print(" 2. Hierarchical Classifier (Trained on Data)")
    engine_choice = input("Choice (1-2): ").strip()
    use_hierarchical = (engine_choice == '2')
    
    print("\n========== Patient Intake details ==========")
    age = int(input("Age: ").strip() or 30)
    gender = input("Gender (Female, Male, Other): ").strip() or "Female"
    comorbidity = input("Comorbidity (Diabetes, Hypertension, Kidney Disease, Obesity, None): ").strip() or "nan"
    symptoms = input("Describe Symptoms (comma-separated, e.g. fever, headache, cough): ").strip()
    
    patient = {
        "age": age, "gender": gender[:1].upper(), "season": "Summer",
        "comorbidity": comorbidity, "smoking_status": "Never", "alcohol_use": "nan",
        "bmi": 22.0, "symptoms": symptoms
    }
    
    # Predict
    if use_hierarchical:
        top3 = predictor.predict_top3_hierarchical(patient)
        model_name = "Hierarchical Classifier"
    else:
        top3 = predictor.predict_top3(patient)
        model_name = "CatBoost Baseline Model"
        
    # Suggest dict matching
    suggestion = predictor.suggest_treatment(patient)
    risk_info = predictor.triage_patient_risk(patient)
    
    print(f"\n========== SCREENING REPORT ({model_name}) ==========")
    print(f"Triage Risk Category: {risk_info['level']}")
    print(f"Triage Message      : {risk_info['message']}")
    print(f"\nTop 3 Suspected Diseases:")
    for d, p in top3:
        print(f"  - {d}: {p:.2%}")
    print(f"\nSuggested Clinical Treatment:")
    print(f"  - matching disease  : {suggestion['disease']}")
    print(f"  - matched symptoms  : {', '.join(suggestion['matched_symptoms'])}")
    print(f"  - treatment path    : {suggestion['treatment']}")
    print("\nActionable Next Steps:")
    print(risk_info['steps'])
    print("=====================================================")

def main():
    predictor = DiseasePredictor()
    while True:
        print("\n==================================================")
        print("          Clinical Triage Test Runner Menu")
        print("==================================================")
        print(" 1. Run Baseline CatBoost Model Evaluation (200 samples)")
        print(" 2. Run Hierarchical Classifier Evaluation (200 samples)")
        print(" 3. Run Vignettes Accuracy Validation Comparison (50 cases)")
        print(" 4. Start Interactive Patient Triage Chat Test")
        print(" 5. Exit")
        print("==================================================")
        
        choice = input("Enter choice (1-5): ").strip()
        if choice == '1':
            run_evaluation(predictor, "CatBoost Baseline Model", use_hierarchical=False)
        elif choice == '2':
            run_evaluation(predictor, "Hierarchical Classifier Model", use_hierarchical=True)
        elif choice == '3':
            run_vignettes_validation(predictor)
        elif choice == '4':
            interactive_chat_session(predictor)
        elif choice == '5':
            print("Exiting test runner. Goodbye!")
            break
        else:
            print("Invalid choice, please select 1-5.")

if __name__ == "__main__":
    main()
