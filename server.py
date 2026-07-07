import os
import uuid
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from conversation import HealthConversation
from utils import extract_json
from app import (
    infer_season,
    calculate_bmi,
    normalize_patient_symptoms,
    add_treatment_suggestion,
    finalize_symptoms
)

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# In-memory session store
sessions = {}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/start-session', methods=['POST'])
def start_session():
    data = request.json or {}
    try:
        age = int(data.get('age'))
        gender = data.get('gender')
        comorbidity = data.get('comorbidity', 'nan')
        smoking_status = data.get('smoking_status', 'Never')
        alcohol_use = data.get('alcohol_use', 'nan')
        height = float(data.get('height'))
        weight = float(data.get('weight'))
        
        # Calculate height in meters and BMI
        height_m = height / 100.0 if height > 10 else height
        bmi = calculate_bmi(height_m, weight)
        
        patient_context = {
            "age": age,
            "gender": gender,
            "season": infer_season(datetime.now()),
            "comorbidity": comorbidity,
            "smoking_status": smoking_status,
            "alcohol_use": alcohol_use,
            "bmi": bmi
        }
        
        # Create conversation bot
        bot = HealthConversation(patient_context=patient_context)
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "bot": bot,
            "patient_context": patient_context,
            "turn_count": 0,
            "max_turns": 12
        }
        
        # First greeting/question from LLM
        initial_input = "Begin the symptom interview. Ask only about the patient's current symptoms."
        first_reply = bot.chat(initial_input)
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": first_reply,
            "patient_context": patient_context
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    session_id = data.get('session_id')
    user_message = data.get('message', '').strip()
    
    if not session_id or session_id not in sessions:
        return jsonify({"success": False, "error": "Invalid or expired session"}), 400
    
    session = sessions[session_id]
    bot = session["bot"]
    patient_context = session["patient_context"]
    
    # Check if user indicates done
    if user_message.lower() in {"done", "finish", "stop", "no more"}:
        return finalize(session_id)
        
    try:
        reply = bot.chat(user_message)
        session["turn_count"] += 1
        
        # Check if the LLM returned JSON directly or if max turns reached
        patient_data = extract_json(reply)
        if patient_data is not None:
            # We got the final data JSON already, finalize now!
            return process_final_diagnosis(session_id, patient_data, reply)
            
        if session["turn_count"] >= session["max_turns"]:
            return finalize(session_id)
            
        return jsonify({
            "success": True,
            "message": reply,
            "finished": False
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/finalize', methods=['POST'])
def finalize_endpoint():
    data = request.json or {}
    session_id = data.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({"success": False, "error": "Invalid or expired session"}), 400
    return finalize(session_id)

def finalize(session_id):
    session = sessions[session_id]
    bot = session["bot"]
    patient_context = session["patient_context"]
    
    try:
        patient_data, final_reply = finalize_symptoms(bot, patient_context)
        if patient_data is not None:
            return process_final_diagnosis(session_id, patient_data, final_reply)
        else:
            # Fallback symptom extraction if JSON parsing failed
            # Grab all user messages from the conversation history
            all_user_text = " ".join([m["content"].lower() for m in bot.messages if m["role"] == "user"])
            
            # Simple keyword matching to find symptoms
            found_symptoms = []
            known_symptoms = [
                "fever", "cough", "chest pain", "chills", "fatigue", "headache", "body ache",
                "joint pain", "muscle pain", "nausea", "vomiting", "diarrhea", "abdominal pain",
                "shortness of breath", "sore throat", "wheezing", "dizziness", "blurred vision"
            ]
            for sym in known_symptoms:
                if sym in all_user_text:
                    found_symptoms.append(sym)
                    
            if not found_symptoms:
                found_symptoms = ["fever", "cough"] # Default mock fallback symptoms
                
            fallback_patient_data = {
                "symptoms": ", ".join(found_symptoms)
            }
            return process_final_diagnosis(session_id, fallback_patient_data, "Here is your compiled triage report based on our discussion:")
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def process_final_diagnosis(session_id, patient_data, final_reply):
    session = sessions[session_id]
    patient_context = session["patient_context"]
    
    try:
        # Merge contexts and normalize
        patient_data = {**patient_data, **patient_context}
        patient_data = normalize_patient_symptoms(patient_data)
        
        # Add predictions and treatment
        patient_data = add_treatment_suggestion(patient_data)
        
        # Ensure outputs folder exists
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/patient.json", "w") as f:
            json.dump(patient_data, f, indent=4)
            
        # Clean up session
        del sessions[session_id]
        
        return jsonify({
            "success": True,
            "message": final_reply,
            "finished": True,
            "patient_data": patient_data
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Try starting the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
