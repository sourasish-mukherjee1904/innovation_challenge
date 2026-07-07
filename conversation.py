import os
import json
import ollama
from groq import Groq

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from prompts import SYSTEM_PROMPT

# Configure Ollama (Primary)
OLLAMA_MODEL = "llama3.1:8b"

# Configure Groq (Fallback)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"


class HealthConversation:

    def __init__(self, patient_context=None):

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        if patient_context:
            self.messages.append(
                {
                    "role": "system",
                    "content": (
                        "Already collected patient context from Python: "
                        f"{json.dumps(patient_context, ensure_ascii=False)}. "
                        "Do not ask about these fields again. Focus only on symptoms."
                    )
                }
            )

    def chat(self, user_input):

        if user_input:
            self.messages.append(
                {
                    "role": "user",
                    "content": user_input
                }
            )

        assistant_reply = None

        # 1. Try Ollama (Primary)
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=self.messages
            )
            assistant_reply = response["message"]["content"]
            print("[+] Query completed successfully using local Ollama.")
        except Exception as e_ollama:
            print(f"[!] Ollama failed or not running ({e_ollama}). Trying Groq fallback...")
            
            # 2. Try Groq (Secondary Fallback)
            if groq_client:
                try:
                    response = groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=self.messages
                    )
                    assistant_reply = response.choices[0].message.content
                    print("[+] Query completed successfully using Groq API.")
                except Exception as e_groq:
                    print(f"[!] Groq query failed ({e_groq}). Using offline mock assistant.")
                    assistant_reply = self._mock_chat_fallback(user_input)
            else:
                print("[!] No Groq API Key found. Using offline mock assistant.")
                assistant_reply = self._mock_chat_fallback(user_input)

        self.messages.append(
            {
                "role": "assistant",
                "content": assistant_reply
            }
        )

        return assistant_reply

    def _mock_chat_fallback(self, user_input):
        user_text = user_input.lower()
        if "begin the symptom interview" in user_text:
            return "Hello! I am the AI Symptom Assistant (Offline Mode). What seems to be your chief complaint today?"
        elif "done" in user_text or "finish" in user_text or "complete" in user_text:
            all_user_text = " ".join([m["content"].lower() for m in self.messages if m["role"] == "user"])
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
                found_symptoms = ["fever", "cough"]
                
            symptoms_str = ", ".join(found_symptoms)
            return f"Thank you. I have recorded your symptoms. Here is the summary:\n```json\n{{\"symptoms\": \"{symptoms_str}\"}}\n```"
        else:
            return "I see. Are you experiencing any other symptoms like fever, cough, joint pain, chest pain, or abdominal pain? Tell me more, or type 'done' to finalize the report."