import pickle
import pandas as pd
from catboost import CatBoostClassifier


class DiseasePredictor:

    def __init__(self):

        # Load CatBoost model
        self.model = CatBoostClassifier()
        self.model.load_model("models/disease_prediction_model.cbm")

        # Load metadata
        with open("models/all_symptoms.pkl", "rb") as f:
            self.all_symptoms = pickle.load(f)

        with open("models/feature_columns.pkl", "rb") as f:
            self.feature_columns = pickle.load(f)

        # Load hierarchical models
        try:
            with open("models/hierarchical/stage1_model.pkl", "rb") as f:
                self.hierarchical_stage1 = pickle.load(f)
            with open("models/hierarchical/stage2_models.pkl", "rb") as f:
                self.hierarchical_stage2 = pickle.load(f)
            self.has_hierarchical = True
        except Exception as e:
            print(f"Hierarchical models not loaded: {e}")
            self.has_hierarchical = False

        self.disease_dictionary = {

            "Respiratory Infection": {
                "symptoms": {
                    "body ache", 
                    "chest pain", 
                    "chills", 
                    "cough", 
                    "fatigue", 
                    "fever", 
                    "loss of appetite", 
                    "loss of smell/taste", 
                    "nausea", 
                    "rash", 
                    "shortness of breath", 
                    "sore throat", 
                    "sweating", 
                    "weight loss", 
                    "wheezing",
                },
                "treatment": "Antibiotics for bacterial cases, oxygen if needed and supportive care.",
            },
            "Mosquito-borne Fever": {
                "symptoms": {
                    "high fever", 
                    "joint pain",
                    "muscle pain",
                    "nausea",
                    "rash",
                    "severe headache",
                    "swollen lymph nodes",
                    "vomiting",
                },
                "treatment": "Fluids and paracetamol, hospitalize for severe cases.",
            },
            "Diarrheal Disease": {
                "symptoms": {
                    "abdominal pain", 
                    "cramps", 
                    "dehydration", 
                    "diarrhea", 
                    "fever", 
                    "nausea", 
                    "vomiting",
                },
                "treatment": "ORS, Zinc for children, fluids if severe.",
            },
            "Chronic Respiratory Disease": {
                "symptoms": {
                    "chest pain", 
                    "cough", 
                    "fatigue", 
                    "fever", 
                    "loss of smell/taste", 
                    "shortness of breath", 
                    "sore throat", 
                    "wheezing",
                },
                "treatment": "Smoking cessation, inhalers and rehab.",
            },
            "Diabetes": {
                "symptoms": {
                    "blurred vision",
                    "chest pain",
                    "dizziness",
                    "fatigue",
                    "frequent urination",
                    "numbness",
                    "shortness of breath",
                    "swelling",
                    "unexplained weight loss",
                },
                "treatment": "Maintain a healthy lifestyle and insulin/medications.",
            },
            "Hypertension": {
                "symptoms": {
                    "blurred vision",
                    "chest pain",
                    "dizziness",
                    "fatigue",
                    "frequent urination",
                    "numbness",
                    "shortness of breath",
                    "swelling",
                    "unexplained weight loss",
                },
                "treatment": "Maintain a healthy lifestyle and take prescribed antihypertensive medications.",
            },
            "Liver Infection": {
                "symptoms": {
                    "abdominal pain",
                    "body ache",
                    "chills",
                    "cramps",
                    "dehydration",
                    "diarrhea",
                    "fatigue",
                    "fever",
                    "loss of appetite",
                    "nausea",
                    "rash",
                    "sweating",
                    "vomiting",
                    "weight loss",
                },
                "treatment": "Supportive care and antivirals when indicated.",
            },
            "Mental Health Disorder": {
                "symptoms": {
                    "appetite changes",
                    "excessive worry",
                    "fatigue",
                    "irritability",
                    "loss of interest",
                    "panic attacks",
                    "persistent sadness",
                    "sleep disturbance"
                },
                "treatment": "Psychotherapy and medication.",
            },
            "Blood Disorder": {
                "symptoms": {
                    "anemia",
                    "bone deformities",
                    "delayed growth",
                    "fatigue",
                    "frequent infections",
                    "pain episodes",
                    "pale skin",
                    "swelling"
                },
                "treatment": "Take iron supplements, eat iron-rich foods, and consult a hematologist for further evaluation.",
            },
            "Chronic Kidney Disease": {
                "symptoms": {
                    "blurred vision",
                    "chest pain",
                    "dizziness",
                    "fatigue",
                    "frequent urination",
                    "numbness",
                    "shortness of breath",
                    "swelling",
                    "unexplained weight loss"
                },
                "treatment": "BP/diabetes control and dialysis if needed.",
            },
            "Leptospirosis": {
                "symptoms": {
                    "body ache",
                    "chills",
                    "fatigue",
                    "fever",
                    "loss of appetite",
                    "nausea",
                    "rash",
                    "sweating",
                    "weight loss"
                },
                "treatment": "Antibiotics and supportive care.",
            },
        }

        self.dictionary_symptom_aliases = {
            "high fever": "fever",
            "severe fever": "fever",
            "body ache": "body ache",
            "body pain": "body ache",
            "muscle ache": "body ache",
            "muscle pain": "body ache",
            "joint pain": "joint pain",
            "head pain": "headache",
            "throat pain": "sore throat",
            "runny nose": "runny nose",
            "nasal congestion": "runny nose",
            "stuffy nose": "runny nose",
            "breathlessness": "shortness of breath",
            "difficulty breathing": "shortness of breath",
            "loss of smell": "loss of smell/taste",
            "loss of taste": "loss of smell/taste",
            "sweats": "sweating",
            "stomach pain": "abdominal pain",
            "belly pain": "abdominal pain",
            "tummy pain": "abdominal pain",
            "vomit": "vomiting",
            "tiredness": "fatigue",
            "weakness": "fatigue",
            "urinating frequently": "frequent urination",
            "peeing frequently": "frequent urination",
            "light sensitivity": "blurred vision",
            "blurred sight": "blurred vision",
            "wheeze": "wheezing",
        }

        self.known_dictionary_symptoms = set()
        for disease_info in self.disease_dictionary.values():
            self.known_dictionary_symptoms.update(disease_info["symptoms"])

        self.canonical_symptoms = {
            symptom.strip().lower(): symptom.strip().lower()
            for symptom in self.all_symptoms
        }

        self.symptom_aliases = {
            "high fever": "fever",
            "severe fever": "fever",
            "fever": "fever",
            "body ache": "body ache",
            "body pain": "body ache",
            "muscle pain": "muscle pain",
            "muscle ache": "muscle pain",
            "joint pain": "joint pain",
            "chest pain": "chest pain",
            "shortness of breath": "shortness of breath",
            "breathlessness": "shortness of breath",
            "sore throat": "sore throat",
            "cough": "cough",
            "rash": "rash",
            "vomiting": "vomiting",
            "nausea": "nausea",
            "diarrhea": "diarrhea",
            "abdominal pain": "abdominal pain",
            "stomach pain": "abdominal pain",
            "dizziness": "dizziness",
            "blurred vision": "blurred vision",
            "fatigue": "fatigue",
            "chills": "chills",
            "sweating": "sweating",
            "swelling": "swelling",
            "swollen lymph nodes": "swollen lymph nodes",
            "loss of appetite": "loss of appetite",
            "appetite changes": "appetite changes",
            "weight loss": "weight loss",
            "severe weight loss": "weight loss",
            "unexplained weight loss": "weight loss",
            "loss of smell": "loss of smell/taste",
            "loss of taste": "loss of smell/taste",
            "loss of smell/taste": "loss of smell/taste",
            "frequent urination": "frequent urination",
            "frequent infections": "frequent infections",
            "panic attacks": "panic attacks",
            "excessive worry": "excessive worry",
            "persistent sadness": "persistent sadness",
            "sleep disturbance": "sleep disturbance",
            "irritability": "irritability",
            "anemia": "anemia",
            "pale skin": "pale skin",
            "delayed growth": "delayed growth",
            "bone deformities": "bone deformities",
            "pain episodes": "pain episodes",
            "cramps": "cramps",
            "dehydration": "dehydration",
            "numbness": "numbness",
            "wheezing": "wheezing",
        }

    def normalize_symptoms(self, raw_symptoms):

        if isinstance(raw_symptoms, list):
            symptom_values = raw_symptoms
        else:
            symptom_values = str(raw_symptoms).split(",")

        normalized = []

        for symptom in symptom_values:
            candidate = str(symptom).strip().lower()

            if not candidate:
                continue

            canonical = self.symptom_aliases.get(candidate, candidate)

            if canonical in self.canonical_symptoms:
                normalized.append(canonical)

        deduped = []

        for symptom in normalized:
            if symptom not in deduped:
                deduped.append(symptom)

        return deduped

    def normalize_dictionary_symptoms(self, raw_symptoms):

        if isinstance(raw_symptoms, list):
            symptom_values = raw_symptoms
        else:
            symptom_values = str(raw_symptoms).split(",")

        normalized = []

        for symptom in symptom_values:
            candidate = str(symptom).strip().lower()

            if not candidate:
                continue

            canonical = self.dictionary_symptom_aliases.get(candidate, candidate)

            if canonical in self.known_dictionary_symptoms:
                normalized.append(canonical)

        deduped = []

        for symptom in normalized:
            if symptom not in deduped:
                deduped.append(symptom)

        return deduped

    def prepare_input(self, patient):

        sample = {}

        # Basic features
        sample["age"] = patient["age"]
        sample["gender"] = patient["gender"]
        sample["season"] = patient["season"]
        sample["comorbidity"] = patient["comorbidity"]
        sample["smoking_status"] = patient["smoking_status"]
        sample["alcohol_use"] = patient["alcohol_use"]
        sample["bmi"] = patient["bmi"]

        # Initialize all symptoms to 0
        for symptom in self.all_symptoms:
            sample[symptom] = 0

        # Mark symptoms present
        raw_symptoms = patient.get("symptoms", "")
        patient_symptoms = self.normalize_symptoms(raw_symptoms)

        for symptom in patient_symptoms:
            if symptom in sample:
                sample[symptom] = 1

        # DataFrame
        sample_df = pd.DataFrame([sample])

        # Ensure correct order
        sample_df = sample_df[self.feature_columns]

        return sample_df

    def predict_top3(self, patient):

        sample_df = self.prepare_input(patient)
        probabilities = self.model.predict_proba(sample_df)[0]
        classes = self.model.classes_

        results = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        return results

    def predict_top3_hierarchical(self, patient):
        if not self.has_hierarchical:
            # Fallback to standard if hierarchical models are not trained/loaded
            return self.predict_top3(patient)

        # 1. Normalize symptoms
        raw_symptoms = patient.get("symptoms", "")
        patient_symptoms = set(self.normalize_symptoms(raw_symptoms))
        
        # 2. Build 44 symptoms binary vector X
        import numpy as np
        X = np.zeros(len(self.all_symptoms))
        for idx, sym in enumerate(self.all_symptoms):
            if sym in patient_symptoms:
                X[idx] = 1.0
                
        # Reshape for sklearn prediction
        X = X.reshape(1, -1)
        
        # 3. Predict Stage 1 categories
        category_probs = self.hierarchical_stage1.predict_proba(X)[0] # Array of 4 probabilities
        
        # 4. Predict Stage 2 diseases
        disease_probs = {}
        
        # Initialize all 12 target classes to 0.0
        for cls in self.model.classes_:
            disease_probs[cls] = 0.0
            
        for cat_id, sub_model in self.hierarchical_stage2.items():
            cat_prob = category_probs[cat_id]
            sub_classes = sub_model.classes_
            sub_probs = sub_model.predict_proba(X)[0]
            
            for sub_cls, sub_p in zip(sub_classes, sub_probs):
                disease_probs[sub_cls] = cat_prob * sub_p
                
        # Sort and return top 3
        results = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        return results




    def suggest_treatment(self, patient):

        raw_symptoms = patient.get("symptoms", "")
        patient_symptoms = set(self.normalize_dictionary_symptoms(raw_symptoms))

        best_match = None

        for disease_name, disease_info in self.disease_dictionary.items():

            disease_symptoms = disease_info["symptoms"]
            matched_symptoms = sorted(patient_symptoms.intersection(disease_symptoms))
            match_count = len(matched_symptoms)

            if match_count == 0:
                continue

            match_ratio = match_count / len(disease_symptoms)

            candidate = {
                "disease": disease_name,
                "matched_symptoms": matched_symptoms,
                "match_count": match_count,
                "match_ratio": round(match_ratio, 3),
                "treatment": disease_info["treatment"],
            }

            if best_match is None or (
                candidate["match_count"],
                candidate["match_ratio"]
            ) > (
                best_match["match_count"],
                best_match["match_ratio"]
            ):

                best_match = candidate

        if best_match is None:

            return {
                "disease": "No clear dictionary match",
                "matched_symptoms": [],
                "match_count": 0,
                "match_ratio": 0.0,
                "treatment": "No disease in the dictionary matched the retrieved symptoms. Review the symptom list with a clinician.",
            }

        return best_match

    def triage_patient_risk(self, patient):
        raw_symptoms = patient.get("symptoms", "")
        normalized_symptoms = set(self.normalize_symptoms(raw_symptoms))
        comorbidity = str(patient.get("comorbidity", "nan")).strip().lower()
        has_comorbidity = comorbidity not in {"nan", "none", "blank", ""}

        # 1. Define High-risk criteria
        high_risk_symptoms = {"shortness of breath", "chest pain", "dehydration", "blurred vision", "numbness"}
        
        # 2. Define Moderate-risk criteria
        moderate_risk_symptoms = {"high fever", "vomiting", "diarrhea", "abdominal pain", "dizziness", "swelling", "wheezing", "fever"}

        # Determine level
        if normalized_symptoms.intersection(high_risk_symptoms):
            level = "High"
        elif normalized_symptoms.intersection(moderate_risk_symptoms):
            # If they have moderate symptoms AND a comorbidity, elevate to High risk
            if has_comorbidity:
                level = "High"
            else:
                level = "Moderate"
        elif has_comorbidity and normalized_symptoms:
            # Comorbidity present even with mild symptoms -> Moderate
            level = "Moderate"
        else:
            level = "Low"

        # Triage messages and actions
        if level == "High":
            message = "Your symptoms and patient profile indicate a potential high-risk medical condition. Urgent evaluation is required."
            steps = (
                "⚠️ **URGENT ACTIONS REQUIRED:**\n"
                "1. **Seek immediate medical care** at the nearest Primary Health Centre (PHC), Community Health Centre (CHC), or emergency facility.\n"
                "2. **Inform medical staff** of any pre-existing conditions, especially Heart Disease, Diabetes, or Kidney Disease.\n"
                "3. **Do not travel alone**; ensure a family member or local ASHA/ANM health worker accompanies you."
            )
        elif level == "Moderate":
            message = "Your symptoms and patient profile suggest a moderate concern. Clinical consultation is recommended soon."
            steps = (
                "⚠️ **RECOMMENDED ACTIONS:**\n"
                "1. **Visit your local healthcare clinic** or consult with your village health worker (ASHA/ANM) within the next 24-48 hours.\n"
                "2. **Keep hydrated** (drink clean boiled water, or ORS if experiencing diarrhea/vomiting) and rest.\n"
                "3. **Monitor your symptoms closely**. If you develop difficulty breathing or chest pain, seek emergency care immediately."
            )
        else:
            message = "Your symptoms appear mild and do not suggest immediate concern. Supportive care is advised."
            steps = (
                "✅ **HOME CARE & MONITORING:**\n"
                "1. **Rest at home** and ensure adequate fluid intake (clean drinking water, juices, coconut water).\n"
                "2. **Supportive measures:** Take paracetamol for mild body aches or fever if appropriate. Avoid taking antibiotics without a prescription.\n"
                "3. **Watch for warning signs:** Contact your ASHA worker if your symptoms worsen or do not improve within 3 days."
            )

        return {
            "level": level,
            "message": message,
            "steps": steps
        }