const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const feedback = document.getElementById("feedback");
const videoFeed = document.getElementById("videoFeed");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const statusValue = document.getElementById("statusValue");
const scoreValue = document.getElementById("scoreValue");
const fatigueValue = document.getElementById("fatigueValue");
const alarmValue = document.getElementById("alarmValue");
const alarmCard = document.getElementById("alarmCard");
const leftEyeValue = document.getElementById("leftEyeValue");
const rightEyeValue = document.getElementById("rightEyeValue");
const eyesDetectedValue = document.getElementById("eyesDetectedValue");
const faceDetectedValue = document.getElementById("faceDetectedValue");

let pollHandle = null;
let audioContext = null;
let alarmTimer = null;

function setFeedback(message, isError = false) {
    feedback.textContent = message;
    feedback.style.color = isError ? "#ff9ea7" : "#a5b6ca";
}

function ensureAudioContext() {
    if (!audioContext) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass) {
            audioContext = new AudioContextClass();
        }
    }

    if (audioContext && audioContext.state === "suspended") {
        audioContext.resume().catch(() => {});
    }
}

function beepOnce() {
    if (!audioContext) {
        return;
    }

    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();

    oscillator.type = "square";
    oscillator.frequency.value = 880;
    gain.gain.setValueAtTime(0.0001, audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.18, audioContext.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioContext.currentTime + 0.28);

    oscillator.connect(gain);
    gain.connect(audioContext.destination);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.3);
}

function startAlarm() {
    ensureAudioContext();
    if (alarmTimer || !audioContext) {
        return;
    }

    beepOnce();
    alarmTimer = window.setInterval(beepOnce, 700);
}

function stopAlarm() {
    if (alarmTimer) {
        window.clearInterval(alarmTimer);
        alarmTimer = null;
    }
}

function renderState(state) {
    statusValue.textContent = state.status;
    scoreValue.textContent = state.score;
    fatigueValue.textContent = state.fatigue || "Standby";
    leftEyeValue.textContent = state.left_eye || "Not detected";
    rightEyeValue.textContent = state.right_eye || "Not detected";
    eyesDetectedValue.textContent = state.eyes_detected ?? 0;
    faceDetectedValue.textContent = state.face_detected ? "Yes" : "No";

    if (state.alarm) {
        alarmValue.textContent = "Alarm On";
        alarmCard.classList.add("alarm");
        startAlarm();
    } else if (state.running) {
        alarmValue.textContent = "Monitoring";
        alarmCard.classList.remove("alarm");
        stopAlarm();
    } else {
        alarmValue.textContent = "Standby";
        alarmCard.classList.remove("alarm");
        stopAlarm();
    }
}

async function refreshStatus() {
    const response = await fetch("/status");
    const state = await response.json();
    renderState(state);
}

function beginPolling() {
    if (pollHandle) {
        window.clearInterval(pollHandle);
    }

    pollHandle = window.setInterval(() => {
        refreshStatus().catch(() => {
            setFeedback("Lost connection while checking status.", true);
            stopAlarm();
        });
    }, 600);
}

function stopPolling() {
    if (pollHandle) {
        window.clearInterval(pollHandle);
        pollHandle = null;
    }
}

startButton.addEventListener("click", async () => {
    startButton.disabled = true;
    ensureAudioContext();
    setFeedback("Starting camera and model...");

    try {
        const response = await fetch("/start", { method: "POST" });
        const result = await response.json();
        renderState(result);

        if (!response.ok || !result.ok) {
            setFeedback(result.message || "Could not start the camera.", true);
            startButton.disabled = false;
            return;
        }

        videoFeed.src = `/video_feed?ts=${Date.now()}`;
        videoFeed.style.display = "block";
        videoPlaceholder.style.display = "none";
        stopButton.disabled = false;
        setFeedback(result.message);
        beginPolling();
    } catch (error) {
        setFeedback("Unable to start detection right now.", true);
        startButton.disabled = false;
    }
});

stopButton.addEventListener("click", async () => {
    stopButton.disabled = true;

    try {
        const response = await fetch("/stop", { method: "POST" });
        const result = await response.json();
        renderState(result);
        stopPolling();
        stopAlarm();
        videoFeed.removeAttribute("src");
        videoFeed.style.display = "none";
        videoPlaceholder.style.display = "grid";
        startButton.disabled = false;
        setFeedback(result.message);
    } catch (error) {
        setFeedback("Could not stop the camera cleanly.", true);
        stopButton.disabled = false;
    }
});

refreshStatus().catch(() => {
    setFeedback("Status could not be loaded yet.", true);
});
