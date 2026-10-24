import sys
import os
import json
import pandas as pd
import numpy as np
from predictor import DiseasePredictor

def resolve_dataset_path():
    # 1. Environment variable override
    env_path = os.environ.get("KAGGLE_DATASET_CSV")
    if env_path and os.path.exists(env_path):
        return env_path

    # 2. Local workspace and system cache candidates
    candidates = [
        "Final_Augmented_dataset_Diseases_and_Symptoms.csv",
        os.path.join("data", "Final_Augmented_dataset_Diseases_and_Symptoms.csv"),
        os.path.expanduser("~/.cache/kagglehub/datasets/dhivyeshrk/diseases-and-symptoms-dataset/versions/1/Final_Augmented_dataset_Diseases_and_Symptoms.csv"),
        r"C:\Users\sayak\.cache\kagglehub\datasets\dhivyeshrk\diseases-and-symptoms-dataset\versions\1\Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # 3. Dynamic kagglehub resolution if installed
    try:
        import kagglehub
        path = kagglehub.dataset_download("dhivyeshrk/diseases-and-symptoms-dataset")
        candidate = os.path.join(path, "Final_Augmented_dataset_Diseases_and_Symptoms.csv")
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass

    return None

def test_vignettes():
    print("==================================================")
    print("Executing Clinical Vignettes Validation (50+ Cases)")
    print("==================================================")
    
    predictor = DiseasePredictor()
    
    data_csv = resolve_dataset_path()
    
    if not data_csv or not os.path.exists(data_csv):
        print("[-] Dataset CSV not found.")
        print("[-] To run evaluations on the Kaggle dataset:")
        print("    1. Set the KAGGLE_DATASET_CSV environment variable: export KAGGLE_DATASET_CSV=/path/to/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
        print("    2. Or download via kagglehub: python -c 'import kagglehub; kagglehub.dataset_download(\"dhivyeshrk/diseases-and-symptoms-dataset\")'")
        print("    3. Or place the CSV file in the project root or data/ directory.")
        sys.exit(1)
        
    df = pd.read_csv(data_csv)
    
    # Target disease mappings
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
    
    # Filter rows that map to our 12 categories
    df_filtered = df[df["diseases"].isin(disease_mapping.keys())].copy()
    
    # Draw 50 random samples (vignettes)
    np.random.seed(101) # constant seed for reproducibility
    df_sample = df_filtered.sample(n=50).reset_index(drop=True)
    
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
    
    cb_correct_top1 = 0
    cb_correct_top3 = 0
    hier_correct_top1 = 0
    hier_correct_top3 = 0
    
    print(f"Simulating screening for {df_sample.shape[0]} clinical vignettes...")
    
    for idx, row in df_sample.iterrows():
        kaggle_disease = row["diseases"]
        expected_class = disease_mapping[kaggle_disease]
        
        # Build symptoms
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
            "age": 30,
            "gender": "F",
            "season": "Summer",
            "comorbidity": "nan",
            "smoking_status": "Never",
            "alcohol_use": "nan",
            "bmi": 22.0,
            "symptoms": ", ".join(present_symptoms) if present_symptoms else "Unknown"
        }
        
        # 1. Run Baseline CatBoost Model
        cb_top3 = predictor.predict_top3(patient)
        cb_classes = [c for c, p in cb_top3]
        if cb_classes and cb_classes[0] == expected_class:
            cb_correct_top1 += 1
        if expected_class in cb_classes:
            cb_correct_top3 += 1
            
        # 2. Run New Hierarchical Classifier (Trained on Data)
        hier_top3 = predictor.predict_top3_hierarchical(patient)
        hier_classes = [c for c, p in hier_top3]
        if hier_classes and hier_classes[0] == expected_class:
            hier_correct_top1 += 1
        if expected_class in hier_classes:
            hier_correct_top3 += 1
            
    total = df_sample.shape[0]
    
    print("\n================ EVALUATION COMPARISON RESULTS ================")
    print(f"Total Test Cases Evaluated : {total}")
    print("\n1. BASELINE CATBOOST MODEL (Clean, No Multipliers):")
    print(f"   - Top-1 Accuracy: {cb_correct_top1 / total:.2%}")
    print(f"   - Top-3 Accuracy: {cb_correct_top3 / total:.2%}")
    
    print("\n2. NEW HIERARCHICAL CLASSIFIER (Trained on Kaggle Data):")
    print(f"   - Top-1 Accuracy: {hier_correct_top1 / total:.2%}")
    print(f"   - Top-3 Accuracy: {hier_correct_top3 / total:.2%}")
    print("===============================================================")

if __name__ == "__main__":
    test_vignettes()
