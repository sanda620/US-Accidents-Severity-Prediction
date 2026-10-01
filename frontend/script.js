const form = document.getElementById("predictionForm");
const result = document.getElementById("result");
const errorBox = document.getElementById("error");
const errorMessage = document.getElementById("errorMessage");
const severityResult = document.getElementById("severityResult");
const descriptionResult = document.getElementById("descriptionResult");
const probabilities = document.getElementById("probabilities");
const predictButton = document.getElementById("predictButton");
const buttonLabel = predictButton.querySelector(".button-label");
const severityEmblem = document.getElementById("severityEmblem");
const resetButton = document.getElementById("resetButton");

// Keep this URL aligned with the address/port where your backend is running.
const API_URL = "http://127.0.0.1:8000/predict";

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  result.classList.add("hidden");
  errorBox.classList.add("hidden");
  errorMessage.textContent = "";
  setLoading(true);

  const data = {
    source: getText("source"),
    street: getText("street"),
    city: getText("city"),
    county: getText("county"),
    state: getText("state"),
    zipcode: getText("zipcode"),
    start_lat: Number(document.getElementById("start_lat").value),
    start_lng: Number(document.getElementById("start_lng").value),
    distance: Number(document.getElementById("distance").value),
    start_time: getText("start_time"),
    temperature: Number(document.getElementById("temperature").value),
    wind_chill: optionalNumber("wind_chill"),
    humidity: Number(document.getElementById("humidity").value),
    pressure: Number(document.getElementById("pressure").value),
    visibility: Number(document.getElementById("visibility").value),
    wind_speed: optionalNumber("wind_speed"),
    precipitation: optionalNumber("precipitation"),
    wind_direction: optionalText("wind_direction"),
    weather_condition: optionalText("weather_condition"),
    airport_code: optionalText("airport_code"),
    timezone: optionalText("timezone"),
    sunrise_sunset: optionalText("sunrise_sunset"),
    civil_twilight: optionalText("civil_twilight"),
    nautical_twilight: optionalText("nautical_twilight"),
    astronomical_twilight: optionalText("astronomical_twilight")
  };

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    let prediction;
    try {
      prediction = await response.json();
    } catch {
      throw new Error("The backend returned an unreadable response. Check that the API is running.");
    }

    if (!response.ok || !prediction.success) {
      throw new Error(prediction.error || `Prediction failed (HTTP ${response.status}).`);
    }

    showResult(prediction);
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    errorMessage.textContent = friendlyError(error);
    errorBox.classList.remove("hidden");
    errorBox.scrollIntoView({ behavior: "smooth", block: "center" });
  } finally {
    setLoading(false);
  }
});

resetButton.addEventListener("click", () => {
  // The reset event occurs after the button's default form reset.
  window.setTimeout(() => {
    result.classList.add("hidden");
    errorBox.classList.add("hidden");
    errorMessage.textContent = "";
  }, 0);
});

function getText(id) {
  return document.getElementById(id).value.trim();
}

function optionalNumber(id) {
  const value = document.getElementById(id).value.trim();
  return value === "" ? null : Number(value);
}

function optionalText(id) {
  const value = getText(id);
  return value === "" ? null : value;
}

function setLoading(isLoading) {
  predictButton.disabled = isLoading;
  predictButton.setAttribute("aria-busy", String(isLoading));
  buttonLabel.textContent = isLoading ? "Analyzing conditions..." : "Predict severity";
  predictButton.querySelector(".button-icon").textContent = isLoading ? "◌" : "✦";
  predictButton.querySelector(".button-arrow").textContent = isLoading ? "…" : "→";
}

function friendlyError(error) {
  const message = error?.message || "An unexpected error occurred.";
  if (error instanceof TypeError && /fetch/i.test(message)) {
    return "Could not connect to the prediction API. Make sure your backend is running at " +
      API_URL + " and that CORS is configured if the frontend is served from another origin.";
  }
  return message;
}

function showResult(prediction) {
  const severity = Number(prediction.severity);
  severityResult.textContent = Number.isFinite(severity)
    ? `Severity ${severity}`
    : "Prediction ready";
  descriptionResult.textContent = prediction.description || "The model has returned a severity prediction.";

  // Set a severity-specific class for visual styling.
  result.classList.remove("severity-1", "severity-2", "severity-3", "severity-4");
  if (severity >= 1 && severity <= 4) {
    result.classList.add(`severity-${severity}`);
  }

  const colors = {
    1: ["#e7faf5", "#1c907e", "#c4eee5"],
    2: ["#edf2ff", "#2856d8", "#dce6ff"],
    3: ["#fff5e5", "#b87515", "#f7e3bd"],
    4: ["#fff0f1", "#c64150", "#f7d7da"]
  };
  const palette = colors[severity] || colors[2];
  severityEmblem.style.background = `linear-gradient(145deg, ${palette[0]}, #ffffff)`;
  severityEmblem.style.color = palette[1];
  severityEmblem.style.borderColor = palette[2];

  probabilities.replaceChildren();
  const entries = Object.entries(prediction.probabilities || {})
    .sort((a, b) => Number(a[0]) - Number(b[0]));

  if (entries.length === 0) {
    const note = document.createElement("p");
    note.className = "result-description";
    note.textContent = "The API did not return class probabilities.";
    probabilities.appendChild(note);
  } else {
    for (const [classLabel, rawProbability] of entries) {
      const probability = Number(rawProbability);
      const safeProbability = Number.isFinite(probability)
        ? Math.min(1, Math.max(0, probability))
        : 0;
      const percent = safeProbability * 100;

      const row = document.createElement("div");
      row.className = `probability-row severity-${Number(classLabel)}`;

      const label = document.createElement("div");
      label.className = "probability-label";
      label.textContent = `Severity ${classLabel}`;

      const track = document.createElement("div");
      track.className = "probability-track";
      track.setAttribute("role", "progressbar");
      track.setAttribute("aria-label", `Severity ${classLabel} probability`);
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", "100");
      track.setAttribute("aria-valuenow", percent.toFixed(2));

      const fill = document.createElement("div");
      fill.className = "probability-fill";
      fill.style.width = "0%";
      track.appendChild(fill);

      const value = document.createElement("div");
      value.className = "probability-value";
      value.textContent = `${percent.toFixed(2)}%`;

      row.append(label, track, value);
      probabilities.appendChild(row);

      // Trigger the bar animation after it has been added to the page.
      requestAnimationFrame(() => {
        fill.style.width = `${percent}%`;
      });
    }
  }

  result.classList.remove("hidden");
}
