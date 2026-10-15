
const API_URL = "http://127.0.0.1:8000";

const form = document.getElementById("predictionForm");
const button = document.getElementById("predictButton");
const statusText = document.getElementById("status");
const result = document.getElementById("result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const data = {};

  for (const [key, value] of formData.entries()) {
    const numericFields = [
      "Hours_Studied",
      "Attendance",
      "Sleep_Hours",
      "Previous_Scores",
      "Tutoring_Sessions",
      "Physical_Activity"
    ];

    data[key] = numericFields.includes(key) ? Number(value) : value;
  }

  button.disabled = true;
  button.textContent = "Predicting...";
  statusText.textContent = "Connecting to the prediction model...";
  result.hidden = true;

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    const body = await response.json();

    if (!response.ok) {
      throw new Error(
        typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail || body)
      );
    }

    document.getElementById("score").textContent =
      body.predicted_exam_score;

    document.getElementById("model").textContent =
      `Model: ${body.model}`;

    const list = document.getElementById("recommendations");
    list.replaceChildren();

    body.recommendations.forEach((recommendation) => {
      const item = document.createElement("li");
      item.textContent = recommendation;
      list.appendChild(item);
    });

    result.hidden = false;
    statusText.textContent = "Prediction completed successfully.";
  } catch (error) {
    statusText.textContent =
      `Prediction failed: ${error.message}. Check that the backend is running.`;
  } finally {
    button.disabled = false;
    button.textContent = "Predict exam score";
  }
});
