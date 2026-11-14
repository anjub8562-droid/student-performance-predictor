const form = document.getElementById("predictForm");
const submitBtn = document.getElementById("submitBtn");
const scoreValue = document.getElementById("scoreValue");
const gaugeProgress = document.getElementById("gaugeProgress");
const tierBadge = document.getElementById("tierBadge");
const insightText = document.getElementById("insightText");

const summaryHours = document.getElementById("summaryHours");
const summaryAttendance = document.getElementById("summaryAttendance");
const summaryPrev = document.getElementById("summaryPrev");

const CIRCUMFERENCE = 2 * Math.PI * 74; // r = 74 -> ~465

const numericFields = [
  "Hours_Studied",
  "Attendance",
  "Sleep_Hours",
  "Previous_Scores",
  "Tutoring_Sessions",
  "Physical_Activity",
];

const presets = {
  balanced: {
    Hours_Studied: 20,
    Attendance: 85,
    Previous_Scores: 70,
    Sleep_Hours: 7,
    Tutoring_Sessions: 1,
    Physical_Activity: 3,
    Parental_Involvement: "High",
    Access_to_Resources: "High",
    Extracurricular_Activities: "No",
    Motivation_Level: "High",
    Internet_Access: "No",
    Family_Income: "High",
    Teacher_Quality: "High",
    School_Type: "Private",
    Peer_Influence: "Negative",
    Learning_Disabilities: "No",
    Parental_Education_Level: "College",
    Distance_from_Home: "Far",
    Gender: "Female",
  },
  achiever: {
    Hours_Studied: 34,
    Attendance: 98,
    Previous_Scores: 92,
    Sleep_Hours: 8,
    Tutoring_Sessions: 4,
    Physical_Activity: 5,
    Parental_Involvement: "High",
    Access_to_Resources: "High",
    Extracurricular_Activities: "Yes",
    Motivation_Level: "High",
    Internet_Access: "Yes",
    Family_Income: "High",
    Teacher_Quality: "High",
    School_Type: "Private",
    Peer_Influence: "Positive",
    Learning_Disabilities: "No",
    Parental_Education_Level: "Postgraduate",
    Distance_from_Home: "Near",
    Gender: "Female",
  },
  atrisk: {
    Hours_Studied: 6,
    Attendance: 62,
    Previous_Scores: 54,
    Sleep_Hours: 5,
    Tutoring_Sessions: 0,
    Physical_Activity: 1,
    Parental_Involvement: "Low",
    Access_to_Resources: "Low",
    Extracurricular_Activities: "No",
    Motivation_Level: "Low",
    Internet_Access: "No",
    Family_Income: "Low",
    Teacher_Quality: "Medium",
    School_Type: "Public",
    Peer_Influence: "Negative",
    Learning_Disabilities: "Yes",
    Parental_Education_Level: "High School",
    Distance_from_Home: "Far",
    Gender: "Male",
  },
};

function syncSliderDisplays() {
  for (const field of numericFields) {
    const input = document.getElementById(field);
    const label = document.getElementById(`${field}_val`);
    if (input && label) {
      label.textContent = input.value;
    }
  }
  summaryHours.textContent = `${document.getElementById("Hours_Studied").value}h`;
  summaryAttendance.textContent = `${document.getElementById("Attendance").value}%`;
  summaryPrev.textContent = document.getElementById("Previous_Scores").value;
}

numericFields.forEach((field) => {
  document.getElementById(field).addEventListener("input", syncSliderDisplays);
});

function updateGauge(score) {
  const clamped = Math.min(100, Math.max(0, score));
  const offset = CIRCUMFERENCE - (clamped / 100) * CIRCUMFERENCE;
  gaugeProgress.style.strokeDashoffset = offset;
  scoreValue.textContent = clamped.toFixed(1);

  tierBadge.className = "tier-badge";
  if (clamped >= 75) {
    document.documentElement.style.setProperty("--tier-color", "#10b981");
    tierBadge.classList.add("tier-high");
    tierBadge.textContent = "🌟 Distinction Track";
    insightText.textContent =
      "Strong study hours and consistent attendance put this profile in the top performance tier.";
  } else if (clamped >= 65) {
    document.documentElement.style.setProperty("--tier-color", "#f59e0b");
    tierBadge.classList.add("tier-mid");
    tierBadge.textContent = "📈 Solid Standing";
    insightText.textContent =
      "Steady baseline performance. Increasing weekly study hours or tutoring sessions can push this past 75.";
  } else {
    document.documentElement.style.setProperty("--tier-color", "#f43f5e");
    tierBadge.classList.add("tier-low");
    tierBadge.textContent = "⚠️ Needs Academic Support";
    insightText.textContent =
      "Low attendance or limited study hours are pulling the score down. Prioritize class attendance first.";
  }
}

async function runPrediction() {
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());
  for (const key of numericFields) {
    payload[key] = Number(payload[key]);
  }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    updateGauge(data.predicted_score);
  } catch (err) {
    console.error(err);
    insightText.textContent =
      "Could not reach /predict endpoint. Make sure uvicorn is running.";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "✨ Predict Exam Score";
  }
}

// Preset buttons
document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document
      .querySelectorAll(".preset-btn")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    const data = presets[btn.dataset.preset];
    Object.entries(data).forEach(([key, val]) => {
      const el = document.getElementById(key);
      if (el) el.value = val;
    });
    syncSliderDisplays();
    runPrediction();
  });
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  runPrediction();
});

// Run initial prediction on load so the gauge animates immediately
syncSliderDisplays();
runPrediction();
