document.addEventListener('DOMContentLoaded', () => {
    // State Variables
    let sessionId = null;
    let patientContext = null;
    const API_BASE = window.location.port !== '5000' ? 'http://127.0.0.1:5000' : '';

    // DOM Elements
    const onboardingStep = document.getElementById('onboarding-step');
    const chatStep = document.getElementById('chat-step');
    const resultStep = document.getElementById('result-step');

    const onboardingForm = document.getElementById('onboarding-form');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const typingIndicator = document.getElementById('typing-indicator');

    const btnStart = document.getElementById('btn-start');
    const btnManualDone = document.getElementById('btn-manual-done');
    const btnRestart = document.getElementById('btn-restart');

    // Result DOM Elements
    const resAge = document.getElementById('res-age');
    const resGender = document.getElementById('res-gender');
    const resBmi = document.getElementById('res-bmi');
    const resSeason = document.getElementById('res-season');
    const resComorbidity = document.getElementById('res-comorbidity');
    const resSmoking = document.getElementById('res-smoking');
    const resAlcohol = document.getElementById('res-alcohol');
    const resDisease = document.getElementById('res-disease');
    const resMatchFill = document.getElementById('res-match-fill');
    const resMatchCount = document.getElementById('res-match-count');
    const resMatchRatio = document.getElementById('res-match-ratio');
    const resSymptoms = document.getElementById('res-symptoms');
    const resTreatment = document.getElementById('res-treatment');

    // Onboarding Form Submit
    onboardingForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        btnStart.disabled = true;
        btnStart.querySelector('span').textContent = 'Starting Session...';
        btnStart.querySelector('i').className = 'fa-solid fa-spinner fa-spin';

        const formData = {
            age: document.getElementById('age').value,
            gender: document.getElementById('gender').value,
            height: document.getElementById('height').value,
            weight: document.getElementById('weight').value,
            comorbidity: document.getElementById('comorbidity').value,
            smoking_status: document.getElementById('smoking_status').value,
            alcohol_use: document.getElementById('alcohol_use').value
        };

        try {
            const response = await fetch(`${API_BASE}/api/start-session`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            if (data.success) {
                sessionId = data.session_id;
                patientContext = data.patient_context;

                // Move to Chat Step
                switchStep(onboardingStep, chatStep);
                
                // Clear previous messages and add the first doctor response
                chatMessages.innerHTML = '';
                appendMessage('bot', data.message);
            } else {
                alert('Error starting session: ' + data.error);
            }
        } catch (error) {
            console.error('API Error:', error);
            alert('Failed to connect to server.');
        } finally {
            // Restore button state
            btnStart.disabled = false;
            btnStart.querySelector('span').textContent = 'Begin Clinical Assessment';
            btnStart.querySelector('i').className = 'fa-solid fa-arrow-right';
        }
    });

    // Chat Form Submit
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const msg = chatInput.value.trim();
        if (!msg) return;

        chatInput.value = '';
        appendMessage('user', msg);

        // Show typing indicator
        showTyping(true);

        try {
            const response = await fetch(`${API_BASE}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, message: msg })
            });

            const data = await response.json();
            showTyping(false);

            if (data.success) {
                if (data.finished) {
                    // Show final report
                    showResults(data.patient_data, data.message);
                } else {
                    appendMessage('bot', data.message);
                    if (data.needs_more_info) {
                        appendMessage('bot', "Please provide a bit more detail about your symptoms so I can finalize the record.");
                    }
                }
            } else {
                appendMessage('bot', 'Error: ' + data.error);
            }
        } catch (error) {
            showTyping(false);
            console.error('API Error:', error);
            appendMessage('bot', 'Failed to communicate with server.');
        }
    });

    // Manual Done Button Click
    btnManualDone.addEventListener('click', async () => {
        if (!sessionId) return;
        
        showTyping(true);
        btnManualDone.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/api/finalize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });

            const data = await response.json();
            showTyping(false);

            if (data.success) {
                if (data.finished) {
                    showResults(data.patient_data, data.message);
                } else {
                    btnManualDone.disabled = false;
                    appendMessage('bot', data.message);
                }
            } else {
                btnManualDone.disabled = false;
                alert('Error finalizing: ' + data.error);
            }
        } catch (error) {
            showTyping(false);
            btnManualDone.disabled = false;
            console.error(error);
            alert('Failed to connect to server.');
        }
    });

    // Restart Button Click
    btnRestart.addEventListener('click', () => {
        sessionId = null;
        patientContext = null;
        onboardingForm.reset();
        btnManualDone.disabled = false;
        switchStep(resultStep, onboardingStep);
    });

    // Helper: Switch Active Card/Step Section
    function switchStep(fromEl, toEl) {
        fromEl.classList.remove('active');
        setTimeout(() => {
            fromEl.style.display = 'none';
            toEl.style.display = 'flex';
            toEl.classList.add('active');
        }, 150);
    }

    // Helper: Append a chat bubble
    function appendMessage(sender, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        bubble.textContent = text;
        
        msgDiv.appendChild(bubble);
        chatMessages.appendChild(msgDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Helper: Toggle Typing Indicator
    function showTyping(show) {
        typingIndicator.style.display = show ? 'flex' : 'none';
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Helper: Populate and display final results
    function showResults(patientData, finalReply) {
        // Switch view to results card
        switchStep(chatStep, resultStep);

        // Fill Patient Profile info
        resAge.textContent = patientData.age || '-';
        resGender.textContent = patientData.gender || '-';
        resBmi.textContent = patientData.bmi || '-';
        resSeason.textContent = patientData.season || '-';
        resComorbidity.textContent = patientData.comorbidity || '-';
        resSmoking.textContent = patientData.smoking_status || '-';
        resAlcohol.textContent = patientData.alcohol_use || '-';

        // Fill Diagnosis & Prediction results
        resDisease.textContent = patientData.suggested_disease || 'Unknown';
        
        const rawRatio = patientData.match_ratio || 0;
        const ratioPercent = (rawRatio * 100).toFixed(1);
        resMatchRatio.textContent = `Confidence: ${ratioPercent}%`;
        resMatchFill.style.width = `${ratioPercent}%`;
        
        const symptomsCount = patientData.matched_symptoms ? patientData.matched_symptoms.length : 0;
        resMatchCount.textContent = `Matched symptoms: ${symptomsCount}`;
        
        resSymptoms.textContent = patientData.symptoms || 'None';
        resTreatment.textContent = patientData.suggested_treatment || 'No specific treatment guidelines matching inputs.';
    }
});
