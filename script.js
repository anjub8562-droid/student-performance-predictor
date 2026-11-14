const form = document.getElementById("predictForm");
const attendanceSlider = document.getElementById("Attendance");
const attendanceVal = document.getElementById("attendanceVal");
const submitBtn = document.getElementById("submitBtn");
const resultCard = document.getElementById("resultCard");
const scoreValue = document.getElementById("scoreValue");
const scoreBar = document.getElementById("scoreBar");

const numericFields = [
  "Hours_Studied",
  "Attendance",
  "Sleep_Hours",
  "Previous_Scores",
  "Tutoring_Sessions",
  "Physical_Activity",
];

// Live slider readout
attendanceSlider.addEventListener("input", () => {
  attendanceVal.textContent = attendanceSlider.value;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = "Calculating...";

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

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const data = await response.json();
    const score = Math.min(100, Math.max(0, data.predicted_score));

    scoreValue.textContent = score.toFixed(1);
    resultCard.classList.remove("hidden");
    requestAnimationFrame(() => {
      scoreBar.style.width = `${score}%`;
    });
  } catch (error) {
    alert("Prediction failed. Check that the backend server is running.");
    console.error(error);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict Exam Score";
  }
});
