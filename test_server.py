import requests
import sys

BASE_URL = "http://127.0.0.1:5000"

def test_health_triage_flow():
    print("==================================================")
    print("Testing CareTriage AI API Endpoints")
    print("==================================================")
    
    # 1. Start Session
    onboarding_data = {
        "age": 32,
        "gender": "Male",
        "height": 172,
        "weight": 68.5,
        "comorbidity": "nan",
        "smoking_status": "Never",
        "alcohol_use": "Occasional"
    }
    
    start_url = f"{BASE_URL}/api/start-session"
    print(f"\n[1/3] POST {start_url} ...")
    try:
        response = requests.post(start_url, json=onboarding_data)
    except requests.exceptions.ConnectionError:
        print(f"[-] Error: Could not connect to Flask server at {BASE_URL}.")
        print("Please make sure server.py is running!")
        sys.exit(1)
        
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers.get('Content-Type')}")
    print(f"Response Body: {response.text}")
    
    if response.status_code != 200:
        print("[-] Start session failed.")
        sys.exit(1)
        
    try:
        data = response.json()
    except Exception as e:
        print(f"[-] Failed to parse response as JSON: {e}")
        sys.exit(1)
        
    if not data.get("success"):
        print(f"[-] Session start returned success=False: {data.get('error')}")
        sys.exit(1)
        
    session_id = data.get("session_id")
    print(f"[+] Session started successfully! ID: {session_id}")
    print(f"Doctor: {data.get('message')}")
    
    # 2. Send Chat Message
    chat_url = f"{BASE_URL}/api/chat"
    chat_payload = {
        "session_id": session_id,
        "message": "I have been experiencing a high fever and severe headache for the past 2 days."
    }
    print(f"\n[2/3] POST {chat_url} ...")
    response = requests.post(chat_url, json=chat_payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
    
    if response.status_code != 200:
        print("[-] Chat request failed.")
        sys.exit(1)
        
    data = response.json()
    print(f"Doctor: {data.get('message')}")
    
    # 3. Finalize Session (Triage & Diagnosis)
    finalize_url = f"{BASE_URL}/api/finalize"
    finalize_payload = {
        "session_id": session_id
    }
    print(f"\n[3/3] POST {finalize_url} ...")
    response = requests.post(finalize_url, json=finalize_payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
    
    if response.status_code != 200:
        print("[-] Finalize request failed.")
        sys.exit(1)
        
    data = response.json()
    if data.get("finished"):
        print("\n[+] Triage Complete!")
        print(f"Target Disease: {data['patient_data'].get('suggested_disease')}")
        print(f"Match Ratio: {data['patient_data'].get('match_ratio')}")
        print(f"Suggested Treatment: {data['patient_data'].get('suggested_treatment')}")
        print("==================================================")
        print("Test Passed successfully!")
    else:
        print("[-] Finalize did not finish correctly.")
        sys.exit(1)

if __name__ == "__main__":
    test_health_triage_flow()
