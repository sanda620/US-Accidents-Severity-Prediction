/**
 * ROADSENSE AI - Traffic Incident Severity Decision Support System
 * Frontend Client Controller
 */

// ==========================================================================
// CONFIGURATION & GLOBAL STATE
// ==========================================================================

let API_URL = localStorage.getItem("roadsense_api_url") || "http://127.0.0.1:8000/predict";
let isBackendOnline = false;

// DOM Elements
const form = document.getElementById("predictionForm");
const predictBtn = document.getElementById("predictButton");
const predictBtnLabel = document.getElementById("predictBtnLabel");
const predictBtnIcon = document.getElementById("predictBtnIcon");
const predictSpinner = document.getElementById("predictSpinner");
const formResetBtn = document.getElementById("formResetBtn");
const quickResetBtn = document.getElementById("quickResetBtn");
const errorCard = document.getElementById("errorCard");
const errorMessage = document.getElementById("errorMessage");
const errorHelpSection = document.getElementById("errorHelpSection");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultCard = document.getElementById("resultCard");
const toastHub = document.getElementById("toastHub");

// Status & Health Elements
const backendStatusPill = document.getElementById("backendStatusPill");
const backendStatusText = document.getElementById("backendStatusText");
const checkHealthBtn = document.getElementById("checkHealthBtn");
const backendOfflineAlert = document.getElementById("backendOfflineAlert");
const offlineUrlDisplay = document.getElementById("offlineUrlDisplay");
const offlineRetryBtn = document.getElementById("offlineRetryBtn");

// Modal Elements
const configEndpointBtn = document.getElementById("configEndpointBtn");
const apiConfigModal = document.getElementById("apiConfigModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const cancelModalBtn = document.getElementById("cancelModalBtn");
const saveModalBtn = document.getElementById("saveModalBtn");
const testApiUrlBtn = document.getElementById("testApiUrlBtn");
const testUrlResult = document.getElementById("testUrlResult");
const apiUrlInput = document.getElementById("apiUrlInput");

// Interactive Distance Elements
const distanceRange = document.getElementById("distanceRange");
const distanceInput = document.getElementById("distance");
const distanceLiveBadge = document.getElementById("distanceLiveBadge");

// Temporal Elements
const startTimeInput = document.getElementById("start_time");
const setNowBtn = document.getElementById("setNowBtn");
const setCurrentTimeQuickBtn = document.getElementById("setCurrentTimeQuickBtn");
const temporalDayText = document.getElementById("temporalDayText");
const temporalRushText = document.getElementById("temporalRushText");
const temporalLightingText = document.getElementById("temporalLightingText");
const temporalCycleText = document.getElementById("temporalCycleText");

// Form Status Badge
const formValidationStatusBadge = document.getElementById("formValidationStatusBadge");
const formValidationStatusText = document.getElementById("formValidationStatusText");
const deckValidationText = document.getElementById("deckValidationText");

// Result Elements
const resultHero = document.getElementById("resultHero");
const severityNumber = document.getElementById("severityNumber");
const severityTitle = document.getElementById("severityTitle");
const severityDescription = document.getElementById("severityDescription");
const severityClassTag = document.getElementById("severityClassTag");
const confidenceText = document.getElementById("confidenceText");
const spectrumMarker = document.getElementById("spectrumMarker");
const probabilitiesList = document.getElementById("probabilitiesList");
const protocolDispatchText = document.getElementById("protocolDispatchText");
const protocolTrafficText = document.getElementById("protocolTrafficText");
const protocolClearanceText = document.getElementById("protocolClearanceText");
const protocolEmergencyText = document.getElementById("protocolEmergencyText");
const riskFactorsTags = document.getElementById("riskFactorsTags");
const copyDispatchReportBtn = document.getElementById("copyDispatchReportBtn");
const scrollToInputsBtn = document.getElementById("scrollToInputsBtn");

// Store latest prediction for report copying
let currentPredictionData = null;
let currentInputData = null;

// ==========================================================================
// SCENARIO PRESETS DATA
// ==========================================================================

const SCENARIO_PRESETS = {
  blizzard: {
    source: "Source2",
    street: "I-80 W",
    city: "Truckee",
    county: "Nevada",
    state: "CA",
    zipcode: "96161",
    start_lat: 39.3279,
    start_lng: -120.1833,
    distance: 8.5,
    start_time: getFormattedDateTime(23, 45, 0),
    timezone: "US/Pacific",
    temperature: 18.0,
    wind_chill: 5.0,
    humidity: 92.0,
    pressure: 28.10,
    visibility: 0.5,
    wind_speed: 35.0,
    precipitation: 0.45,
    wind_direction: "NW",
    weather_condition: "Heavy Snow",
    airport_code: "KTRK",
    sunrise_sunset: "Night",
    civil_twilight: "Night",
    nautical_twilight: "Night",
    astronomical_twilight: "Night",
    toastMsg: "Loaded Scenario: Severe Mountain Blizzard (Expected Level 4)"
  },
  "corridor-pileup": {
    source: "Source2",
    street: "I-5 N",
    city: "Los Angeles",
    county: "Los Angeles",
    state: "CA",
    zipcode: "90012",
    start_lat: 34.0522,
    start_lng: -118.2437,
    distance: 0.0,
    start_time: "2017-08-15T08:15",
    timezone: "US/Pacific",
    temperature: 65.0,
    wind_chill: null,
    humidity: 70.0,
    pressure: 29.95,
    visibility: 5.0,
    wind_speed: 10.0,
    precipitation: 0.1,
    wind_direction: "SW",
    weather_condition: "Light Rain",
    airport_code: "KLAX",
    sunrise_sunset: "Day",
    civil_twilight: "Day",
    nautical_twilight: "Day",
    astronomical_twilight: "Day",
    toastMsg: "Loaded Scenario: Major Interstate Corridor Disruption (Predicted Level 3)"
  },
  "highway-rain": {
    source: "Source2",
    street: "I-5 N",
    city: "Los Angeles",
    county: "Los Angeles",
    state: "CA",
    zipcode: "90012",
    start_lat: 34.0522,
    start_lng: -118.2437,
    distance: 0.0,
    start_time: "2017-08-15T08:15",
    timezone: "US/Pacific",
    temperature: 65.0,
    wind_chill: null,
    humidity: 70.0,
    pressure: 29.95,
    visibility: 5.0,
    wind_speed: 10.0,
    precipitation: 0.1,
    wind_direction: "SW",
    weather_condition: "Light Rain",
    airport_code: "KLAX",
    sunrise_sunset: "Day",
    civil_twilight: "Day",
    nautical_twilight: "Day",
    astronomical_twilight: "Day",
    toastMsg: "Loaded Scenario: Major Interstate Corridor Disruption (Predicted Level 3)"
  },
  arterial: {
    source: "Source2",
    street: "Broad St",
    city: "Philadelphia",
    county: "Philadelphia",
    state: "PA",
    zipcode: "19107",
    start_lat: 39.9526,
    start_lng: -75.1652,
    distance: 0.4,
    start_time: getFormattedDateTime(8, 30, 0),
    timezone: "US/Eastern",
    temperature: 58.0,
    wind_chill: null,
    humidity: 62.0,
    pressure: 30.05,
    visibility: 10.0,
    wind_speed: 8.0,
    precipitation: 0.0,
    wind_direction: "W",
    weather_condition: "Overcast",
    airport_code: "KPHL",
    sunrise_sunset: "Day",
    civil_twilight: "Day",
    nautical_twilight: "Day",
    astronomical_twilight: "Day",
    toastMsg: "Loaded Scenario: Moderate Urban Arterial Rush Incident"
  },
  minor: {
    source: "Source2",
    street: "Main St",
    city: "Austin",
    county: "Travis",
    state: "TX",
    zipcode: "78701",
    start_lat: 30.2672,
    start_lng: -97.7431,
    distance: 0.05,
    start_time: getFormattedDateTime(14, 15, 0),
    timezone: "US/Central",
    temperature: 77.0,
    wind_chill: null,
    humidity: 50.0,
    pressure: 29.98,
    visibility: 10.0,
    wind_speed: 5.0,
    precipitation: 0.0,
    wind_direction: "NE",
    weather_condition: "Clear",
    airport_code: "KAUS",
    sunrise_sunset: "Day",
    civil_twilight: "Day",
    nautical_twilight: "Day",
    astronomical_twilight: "Day",
    toastMsg: "Loaded Scenario: Minor Local Street Bump"
  }
};

const METRO_PRESETS = {
  miami: { street: "I-95 S", city: "Miami", county: "Miami-Dade", state: "FL", zipcode: "33101", lat: 25.7617, lng: -80.1918, airport: "KMIA", tz: "US/Eastern" },
  la: { street: "I-405 N", city: "Los Angeles", county: "Los Angeles", state: "CA", zipcode: "90012", lat: 34.0522, lng: -118.2437, airport: "KLAX", tz: "US/Pacific" },
  chicago: { street: "I-90 E", city: "Chicago", county: "Cook", state: "IL", zipcode: "60601", lat: 41.8781, lng: -87.6298, airport: "KORD", tz: "US/Central" },
  nyc: { street: "FDR Dr", city: "New York", county: "New York", state: "NY", zipcode: "10001", lat: 40.7128, lng: -74.0060, airport: "KJFK", tz: "US/Eastern" }
};

const WEATHER_PRESETS = {
  clear: { temp: 72.0, chill: null, humidity: 45.0, pressure: 29.95, vis: 10.0, wind: 6.0, precip: 0.0, cond: "Clear" },
  rain: { temp: 64.0, chill: null, humidity: 90.0, pressure: 29.75, vis: 3.5, wind: 16.0, precip: 0.20, cond: "Rain" },
  snow: { temp: 24.0, chill: 12.0, humidity: 88.0, pressure: 29.60, vis: 1.5, wind: 24.0, precip: 0.35, cond: "Snow" },
  fog: { temp: 52.0, chill: null, humidity: 98.0, pressure: 29.90, vis: 0.4, wind: 3.0, precip: 0.0, cond: "Fog" }
};

// ==========================================================================
// INITIALIZATION
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Set default current time
  setCurrentTime();

  // Initial backend health check
  checkBackendHealth();

  // Setup periodic health monitoring
  setInterval(checkBackendHealth, 30000);

  // Bind all event listeners
  bindEvents();

  // Initial validation check & temporal calculation
  updateDistanceFeedback();
  calculateTemporalFeatures();
  validateFormInputs(false);
});

// ==========================================================================
// EVENT BINDINGS
// ==========================================================================

function bindEvents() {
  // Distance range <-> input sync
  distanceRange.addEventListener("input", (e) => {
    distanceInput.value = parseFloat(e.target.value).toFixed(2);
    updateDistanceFeedback();
    validateFormInputs(false);
  });

  distanceInput.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    if (!isNaN(val)) {
      distanceRange.value = Math.min(20, Math.max(0, val));
    }
    updateDistanceFeedback();
    validateFormInputs(false);
  });

  // Timepicker change updates temporal intelligence
  startTimeInput.addEventListener("input", calculateTemporalFeatures);

  // Time buttons
  setNowBtn.addEventListener("click", () => {
    setCurrentTime();
    showToast("Incident timestamp set to current time", "info");
  });

  setCurrentTimeQuickBtn.addEventListener("click", () => {
    setCurrentTime();
    showToast("Incident timestamp updated", "info");
  });

  // Scenario quick presets
  document.querySelectorAll("[data-scenario]").forEach(btn => {
    btn.addEventListener("click", () => {
      const scenarioKey = btn.getAttribute("data-scenario");
      loadScenario(scenarioKey);
    });
  });

  // Metro presets
  document.querySelectorAll("[data-metro]").forEach(btn => {
    btn.addEventListener("click", () => {
      const metroKey = btn.getAttribute("data-metro");
      loadMetroPreset(metroKey);
    });
  });

  // Weather presets
  document.querySelectorAll("[data-weather]").forEach(btn => {
    btn.addEventListener("click", () => {
      const weatherKey = btn.getAttribute("data-weather");
      loadWeatherPreset(weatherKey);
    });
  });

  // Twilight buttons
  document.getElementById("setAllDayBtn").addEventListener("click", () => {
    setAllTwilight("Day");
  });

  document.getElementById("setAllNightBtn").addEventListener("click", () => {
    setAllTwilight("Night");
  });

  // Form submission
  form.addEventListener("submit", handleFormSubmit);

  // Form reset buttons
  formResetBtn.addEventListener("click", resetAllInputs);
  quickResetBtn.addEventListener("click", resetAllInputs);

  // Health check button & retry
  checkHealthBtn.addEventListener("click", () => {
    checkBackendHealth(true);
  });

  offlineRetryBtn.addEventListener("click", () => {
    checkBackendHealth(true);
  });

  // Modal events
  configEndpointBtn.addEventListener("click", openConfigModal);
  closeModalBtn.addEventListener("click", closeConfigModal);
  cancelModalBtn.addEventListener("click", closeConfigModal);
  saveModalBtn.addEventListener("click", saveEndpointConfig);
  testApiUrlBtn.addEventListener("click", testEndpointInModal);

  // Close modal on click outside
  apiConfigModal.addEventListener("click", (e) => {
    if (e.target === apiConfigModal) closeConfigModal();
  });

  // Action buttons
  copyDispatchReportBtn.addEventListener("click", copyDispatchReportToClipboard);
  scrollToInputsBtn.addEventListener("click", () => {
    form.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  // Real-time input validation on change/blur
  const inputsToValidate = form.querySelectorAll("input, select");
  inputsToValidate.forEach(input => {
    input.addEventListener("input", () => validateField(input));
    input.addEventListener("blur", () => validateField(input));
  });
}

// ==========================================================================
// BACKEND CONNECTION & HEALTH CHECK
// ==========================================================================

async function checkBackendHealth(showFeedback = false) {
  const healthUrl = API_URL.replace(/\/predict\/?$/, "/health");
  const startTime = performance.now();

  backendStatusText.textContent = "Pinging...";

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const response = await fetch(healthUrl, {
      method: "GET",
      signal: controller.signal
    });

    clearTimeout(timeoutId);
    const latency = Math.round(performance.now() - startTime);

    if (response.ok) {
      const data = await response.json();
      isBackendOnline = true;
      backendStatusPill.className = "status-indicator online";
      backendStatusText.textContent = `API Online (${latency}ms · XGBoost)`;
      backendOfflineAlert.classList.add("hidden");

      if (showFeedback) {
        showToast(`Backend connection healthy (${latency}ms)`, "success");
      }
      return true;
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (err) {
    isBackendOnline = false;
    backendStatusPill.className = "status-indicator offline";
    backendStatusText.textContent = "Backend Offline";
    offlineUrlDisplay.textContent = API_URL;
    backendOfflineAlert.classList.remove("hidden");

    if (showFeedback) {
      showToast("Could not reach backend API server", "error");
    }
    return false;
  }
}

function openConfigModal() {
  apiUrlInput.value = API_URL;
  testUrlResult.textContent = "";
  testUrlResult.className = "test-result-text";
  apiConfigModal.classList.remove("hidden");
}

function closeConfigModal() {
  apiConfigModal.classList.add("hidden");
}

async function testEndpointInModal() {
  const targetUrl = apiUrlInput.value.trim();
  const healthUrl = targetUrl.replace(/\/predict\/?$/, "/health");

  testUrlResult.textContent = "Testing...";
  testUrlResult.className = "test-result-text";

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3500);

    const response = await fetch(healthUrl, {
      method: "GET",
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (response.ok) {
      testUrlResult.textContent = "✓ Connected Successfully";
      testUrlResult.className = "test-result-text success";
    } else {
      testUrlResult.textContent = `✗ HTTP Error ${response.status}`;
      testUrlResult.className = "test-result-text fail";
    }
  } catch (err) {
    testUrlResult.textContent = "✗ Connection Refused";
    testUrlResult.className = "test-result-text fail";
  }
}

function saveEndpointConfig() {
  const newUrl = apiUrlInput.value.trim();
  if (!newUrl) {
    showToast("Please provide a valid URL", "error");
    return;
  }

  API_URL = newUrl;
  localStorage.setItem("roadsense_api_url", API_URL);
  closeConfigModal();
  showToast("API Endpoint URL updated", "success");
  checkBackendHealth(true);
}

// ==========================================================================
// PRESETS & SCENARIO ENGINE
// ==========================================================================

function loadScenario(key) {
  const data = SCENARIO_PRESETS[key];
  if (!data) return;

  // Set all values
  setVal("source", data.source);
  setVal("street", data.street);
  setVal("city", data.city);
  setVal("county", data.county);
  setVal("state", data.state);
  setVal("zipcode", data.zipcode);
  setVal("start_lat", data.start_lat);
  setVal("start_lng", data.start_lng);
  setVal("distance", data.distance);
  distanceRange.value = Math.min(20, data.distance);

  setVal("start_time", data.start_time);
  setVal("timezone", data.timezone);

  setVal("temperature", data.temperature);
  setVal("wind_chill", data.wind_chill !== null ? data.wind_chill : "");
  setVal("humidity", data.humidity);
  setVal("pressure", data.pressure);
  setVal("visibility", data.visibility);
  setVal("wind_speed", data.wind_speed !== null ? data.wind_speed : "");
  setVal("precipitation", data.precipitation !== null ? data.precipitation : "");
  setVal("wind_direction", data.wind_direction);
  setVal("weather_condition", data.weather_condition);

  setVal("airport_code", data.airport_code);
  setVal("sunrise_sunset", data.sunrise_sunset);
  setVal("civil_twilight", data.civil_twilight);
  setVal("nautical_twilight", data.nautical_twilight);
  setVal("astronomical_twilight", data.astronomical_twilight);

  updateDistanceFeedback();
  calculateTemporalFeatures();
  validateFormInputs(false);

  showToast(data.toastMsg || "Scenario loaded", "info");
}

function loadMetroPreset(key) {
  const metro = METRO_PRESETS[key];
  if (!metro) return;

  setVal("street", metro.street);
  setVal("city", metro.city);
  setVal("county", metro.county);
  setVal("state", metro.state);
  setVal("zipcode", metro.zipcode);
  setVal("start_lat", metro.lat);
  setVal("start_lng", metro.lng);
  setVal("airport_code", metro.airport);
  setVal("timezone", metro.tz);

  validateFormInputs(false);
  showToast(`Location set to ${metro.city}, ${metro.state}`, "info");
}

function loadWeatherPreset(key) {
  const w = WEATHER_PRESETS[key];
  if (!w) return;

  setVal("temperature", w.temp);
  setVal("wind_chill", w.chill !== null ? w.chill : "");
  setVal("humidity", w.humidity);
  setVal("pressure", w.pressure);
  setVal("visibility", w.vis);
  setVal("wind_speed", w.wind);
  setVal("precipitation", w.precip);
  setVal("weather_condition", w.cond);

  validateFormInputs(false);
  showToast(`Applied ${w.cond} weather preset`, "info");
}

function setAllTwilight(val) {
  setVal("sunrise_sunset", val);
  setVal("civil_twilight", val);
  setVal("nautical_twilight", val);
  setVal("astronomical_twilight", val);
  showToast(`Twilight & solar horizon set to ${val}`, "info");
}

function resetAllInputs() {
  form.reset();
  setCurrentTime();
  setVal("source", "Source2");
  setVal("street", "I-95 S");
  setVal("city", "Miami");
  setVal("county", "Miami-Dade");
  setVal("state", "FL");
  setVal("zipcode", "33101");
  setVal("start_lat", 25.7617);
  setVal("start_lng", -80.1918);
  setVal("distance", 1.5);
  distanceRange.value = 1.5;
  setVal("temperature", 72.0);
  setVal("humidity", 65);
  setVal("pressure", 29.92);
  setVal("visibility", 10.0);
  setVal("wind_speed", 12.0);
  setVal("precipitation", 0.0);

  updateDistanceFeedback();
  calculateTemporalFeatures();
  validateFormInputs(false);

  // Hide errors and results
  errorCard.classList.add("hidden");
  resultCard.classList.add("hidden");
  resultPlaceholder.classList.remove("hidden");

  showToast("Form parameters reset to standard baseline", "info");
}

// ==========================================================================
// TEMPORAL DYNAMICS & FEEDBACK
// ==========================================================================

function setCurrentTime() {
  const now = new Date();
  startTimeInput.value = formatDateTimeLocal(now);
  calculateTemporalFeatures();
}

function formatDateTimeLocal(date) {
  const pad = (n) => String(n).padStart(2, "0");
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  return `${year}-${month}-${day}T${hours}:${minutes}`;
}

function getFormattedDateTime(hours, minutes, daysAgo = 0) {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  d.setHours(hours, minutes, 0, 0);
  return formatDateTimeLocal(d);
}

function calculateTemporalFeatures() {
  const timeVal = startTimeInput.value;
  if (!timeVal) {
    temporalDayText.textContent = "—";
    temporalRushText.textContent = "—";
    temporalLightingText.textContent = "—";
    return;
  }

  const date = new Date(timeVal);
  if (isNaN(date.getTime())) return;

  const days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  const dayName = days[date.getDay()];
  const isWeekend = date.getDay() === 0 || date.getDay() === 6;

  const hour = date.getHours();
  const isMorningRush = hour >= 7 && hour <= 9;
  const isEveningRush = hour >= 16 && hour <= 19;
  const isNight = hour < 6 || hour >= 20;

  let trafficProfile = "Standard Flow";
  if (isMorningRush) trafficProfile = "Morning Peak (Rush)";
  else if (isEveningRush) trafficProfile = "Evening Peak (Rush)";
  else if (isWeekend) trafficProfile = "Weekend Moderate";

  temporalDayText.textContent = `${dayName} (${isWeekend ? "Weekend" : "Weekday"})`;
  temporalRushText.textContent = trafficProfile;
  temporalLightingText.textContent = isNight ? "Night / Darkness" : "Daylight Hours";

  // Auto-sync twilight if user hasn't explicitly customized
  if (isNight && document.getElementById("sunrise_sunset").value === "Day") {
    setVal("sunrise_sunset", "Night");
    setVal("civil_twilight", "Night");
  } else if (!isNight && document.getElementById("sunrise_sunset").value === "Night") {
    setVal("sunrise_sunset", "Day");
    setVal("civil_twilight", "Day");
  }
}

function updateDistanceFeedback() {
  const dist = parseFloat(distanceInput.value) || 0;
  let label = "Localized Impact (<0.5 mi)";

  if (dist >= 5.0) {
    label = `${dist.toFixed(1)} mi (Critical Corridor Blockage)`;
  } else if (dist >= 2.0) {
    label = `${dist.toFixed(1)} mi (Major Highway Backup)`;
  } else if (dist >= 0.5) {
    label = `${dist.toFixed(1)} mi (Moderate Queue)`;
  } else {
    label = `${dist.toFixed(2)} mi (Localized)`;
  }

  distanceLiveBadge.textContent = label;
}

// ==========================================================================
// VALIDATION ENGINE
// ==========================================================================

const VALIDATION_RULES = {
  start_lat: { min: -90, max: 90, errId: "latError", name: "Latitude" },
  start_lng: { min: -180, max: 180, errId: "lngError", name: "Longitude" },
  distance: { min: 0, max: 200, errId: "distanceError", name: "Distance" },
  temperature: { min: -80, max: 140, errId: "tempError", name: "Temperature" },
  wind_chill: { min: -100, max: 150, errId: "chillError", name: "Wind Chill", optional: true },
  humidity: { min: 0, max: 100, errId: "humidityError", name: "Humidity" },
  pressure: { min: 20, max: 35, errId: "pressureError", name: "Atmospheric Pressure" },
  visibility: { min: 0, max: 100, errId: "visibilityError", name: "Visibility" },
  wind_speed: { min: 0, max: 100, errId: "windSpeedError", name: "Wind Speed", optional: true },
  precipitation: { min: 0, max: 50, errId: "precipError", name: "Precipitation", optional: true }
};

function validateField(input) {
  const id = input.id;
  const val = input.value.trim();

  // Range validation for numeric fields
  if (VALIDATION_RULES[id]) {
    const rule = VALIDATION_RULES[id];
    const errEl = document.getElementById(rule.errId);

    if (val === "" && rule.optional) {
      input.classList.remove("is-invalid");
      input.classList.remove("is-valid");
      if (errEl) errEl.classList.remove("visible");
      return true;
    }

    const num = parseFloat(val);
    const isValid = !isNaN(num) && num >= rule.min && num <= rule.max;

    if (!isValid) {
      input.classList.add("is-invalid");
      input.classList.remove("is-valid");
      if (errEl) {
        errEl.textContent = `${rule.name} must be between ${rule.min} and ${rule.max}`;
        errEl.classList.add("visible");
      }
      return false;
    } else {
      input.classList.remove("is-invalid");
      input.classList.add("is-valid");
      if (errEl) errEl.classList.remove("visible");
      return true;
    }
  }

  // Required text validation
  if (input.required) {
    const errEl = document.getElementById(`${id}Error`);
    if (val === "") {
      input.classList.add("is-invalid");
      input.classList.remove("is-valid");
      if (errEl) errEl.classList.add("visible");
      return false;
    } else {
      input.classList.remove("is-invalid");
      input.classList.add("is-valid");
      if (errEl) errEl.classList.remove("visible");
      return true;
    }
  }

  return true;
}

function validateFormInputs(showErrors = true) {
  let isAllValid = true;
  let firstInvalidEl = null;

  // Check required text & select fields
  const requiredIds = ["street", "city", "county", "state", "zipcode", "start_time", "source"];
  requiredIds.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    const val = el.value.trim();
    const errEl = document.getElementById(`${id}Error`);

    if (val === "") {
      isAllValid = false;
      if (showErrors) {
        el.classList.add("is-invalid");
        if (errEl) errEl.classList.add("visible");
        if (!firstInvalidEl) firstInvalidEl = el;
      }
    } else {
      el.classList.remove("is-invalid");
      if (errEl) errEl.classList.remove("visible");
    }
  });

  // Check numeric rules
  Object.keys(VALIDATION_RULES).forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    const isValid = validateField(el);
    if (!isValid) {
      isAllValid = false;
      if (!firstInvalidEl) firstInvalidEl = el;
    }
  });

  // Update top validation badge
  if (isAllValid) {
    formValidationStatusBadge.className = "form-validator-badge";
    formValidationStatusBadge.innerHTML = `<span class="validator-icon">✓</span> <span class="validator-text">All Required Inputs Valid</span>`;
    deckValidationText.textContent = "Form parameters fully specified for XGBoost evaluation";
  } else {
    formValidationStatusBadge.className = "form-validator-badge has-errors";
    formValidationStatusBadge.innerHTML = `<span class="validator-icon">!</span> <span class="validator-text">Inputs Require Attention</span>`;
    deckValidationText.textContent = "Please correct flagged fields above before predicting";
  }

  return { isValid: isAllValid, firstInvalid: firstInvalidEl };
}

// ==========================================================================
// PREDICTION WORKFLOW & INFERENCE
// ==========================================================================

async function handleFormSubmit(event) {
  event.preventDefault();

  // Pre-submission validation
  const validation = validateFormInputs(true);
  if (!validation.isValid) {
    if (validation.firstInvalid) {
      validation.firstInvalid.scrollIntoView({ behavior: "smooth", block: "center" });
      validation.firstInvalid.focus();
    }
    showToast("Please fix invalid or missing fields before proceeding", "error");
    return;
  }

  // Hide previous errors & show loading
  errorCard.classList.add("hidden");
  setLoadingState(true);

  // Construct payload matching backend AccidentInput schema
  const payload = {
    source: getVal("source"),
    street: getVal("street"),
    city: getVal("city"),
    county: getVal("county"),
    state: getVal("state"),
    zipcode: getVal("zipcode"),
    start_lat: parseFloat(getVal("start_lat")),
    start_lng: parseFloat(getVal("start_lng")),
    distance: parseFloat(getVal("distance")),
    start_time: getVal("start_time"),
    temperature: parseFloat(getVal("temperature")),
    wind_chill: getOptionalNumber("wind_chill"),
    humidity: parseFloat(getVal("humidity")),
    pressure: parseFloat(getVal("pressure")),
    visibility: parseFloat(getVal("visibility")),
    wind_speed: getOptionalNumber("wind_speed"),
    precipitation: getOptionalNumber("precipitation"),
    wind_direction: getOptionalText("wind_direction"),
    weather_condition: getOptionalText("weather_condition"),
    airport_code: getOptionalText("airport_code"),
    timezone: getOptionalText("timezone"),
    sunrise_sunset: getOptionalText("sunrise_sunset"),
    civil_twilight: getOptionalText("civil_twilight"),
    nautical_twilight: getOptionalText("nautical_twilight"),
    astronomical_twilight: getOptionalText("astronomical_twilight")
  };

  currentInputData = payload;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    let result;
    try {
      result = await response.json();
    } catch {
      throw new Error("Received malformed or non-JSON response from prediction server.");
    }

    if (!response.ok || !result.success) {
      throw new Error(result.error || `Prediction request failed with status HTTP ${response.status}`);
    }

    // Success: Render results
    currentPredictionData = result;
    renderPredictionResults(result, payload);
    showToast(`Severity Level ${result.severity} Predicted`, "success");

    // Scroll to result on smaller screens
    if (window.innerWidth <= 1200) {
      resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (err) {
    handlePredictionError(err);
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(isLoading) {
  predictBtn.disabled = isLoading;
  if (isLoading) {
    predictSpinner.classList.remove("hidden");
    predictBtnIcon.classList.add("hidden");
    predictBtnLabel.textContent = "Computing XGBoost Severity...";
  } else {
    predictSpinner.classList.add("hidden");
    predictBtnIcon.classList.remove("hidden");
    predictBtnLabel.textContent = "Predict Severity Level";
  }
}

function handlePredictionError(err) {
  const msg = err.message || "An unexpected error occurred.";
  errorMessage.textContent = msg;

  if (err instanceof TypeError && /fetch/i.test(msg)) {
    errorHelpSection.innerHTML = `
      <strong>Backend Server Offline or Unreachable:</strong>
      Ensure the FastAPI backend is running with:
      <code>.\\venv\\Scripts\\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000</code>
    `;
    checkBackendHealth();
  } else {
    errorHelpSection.innerHTML = `
      Verify that input parameter ranges conform to dataset boundaries.
    `;
  }

  errorCard.classList.remove("hidden");
  errorCard.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ==========================================================================
// RESULTS RENDERING & DECISION SUPPORT PROTOCOLS
// ==========================================================================

function renderPredictionResults(prediction, inputs) {
  const severity = parseInt(prediction.severity, 10) || 2;
  const probs = prediction.probabilities || {};

  // Find winning probability
  const winningProb = (probs[String(severity)] || 0) * 100;

  // 1. Update Hero Card Classes
  resultCard.className = `result-card is-sev${severity}`;
  severityNumber.textContent = severity;
  severityTitle.textContent = `Severity Level ${severity}`;
  severityClassTag.textContent = `LEVEL ${severity} CLASSIFICATION`;
  severityDescription.textContent = prediction.description || getSeverityOperationalSummary(severity);
  confidenceText.textContent = `Model Confidence: ${winningProb.toFixed(1)}%`;

  // 2. Position Spectrum Marker
  // Levels: 1 (12.5%), 2 (37.5%), 3 (62.5%), 4 (87.5%)
  const markerPositions = { 1: "12.5%", 2: "37.5%", 3: "62.5%", 4: "87.5%" };
  spectrumMarker.style.left = markerPositions[severity] || "37.5%";

  // 3. Render Probability Breakdown Bars
  renderProbabilityBars(probs, severity);

  // 4. Render Operational Dispatch Protocols
  renderOperationalProtocols(severity, inputs);

  // 5. Render Detected Aggravating Risk Factors
  renderRiskFactors(inputs, severity);

  // Switch from placeholder to active card
  resultPlaceholder.classList.add("hidden");
  resultCard.classList.remove("hidden");
}

function getSeverityOperationalSummary(severity) {
  switch (severity) {
    case 1:
      return "Minor impact incident. Negligible road disruption; localized shoulder or single vehicle incident.";
    case 2:
      return "Moderate impact accident with notable localized traffic delay and partial road obstruction.";
    case 3:
      return "Significant impact incident causing major traffic backups, multi-lane closure, and prolonged clearance.";
    case 4:
      return "Critical / Severe impact incident causing full highway blockage, major gridlock, and emergency response escalation.";
    default:
      return "Estimated traffic-impact severity.";
  }
}

function renderProbabilityBars(probabilities, winningSeverity) {
  probabilitiesList.replaceChildren();

  const classes = [1, 2, 3, 4];
  classes.forEach(cls => {
    const rawVal = probabilities[String(cls)] ?? 0;
    const pct = Math.min(100, Math.max(0, rawVal * 100));
    const isWinner = cls === winningSeverity;

    const row = document.createElement("div");
    row.className = `prob-row class-${cls}`;

    const label = document.createElement("div");
    label.className = "prob-label";
    label.innerHTML = `Level ${cls} ${isWinner ? '<span class="prob-winner-tag">WIN</span>' : ""}`;

    const track = document.createElement("div");
    track.className = "prob-track";

    const fill = document.createElement("div");
    fill.className = "prob-fill";
    fill.style.width = "0%";
    track.appendChild(fill);

    const val = document.createElement("div");
    val.className = "prob-val";
    val.textContent = `${pct.toFixed(2)}%`;

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(val);
    probabilitiesList.appendChild(row);

    // Trigger smooth fill animation
    requestAnimationFrame(() => {
      fill.style.width = `${pct}%`;
    });
  });
}

function renderOperationalProtocols(severity, inputs) {
  if (severity === 1) {
    protocolDispatchText.textContent = "Standard municipal patrol unit. Localized roadside assistance or private towing.";
    protocolTrafficText.textContent = "No detour necessary. Standard advisory hazard blinkers.";
    protocolClearanceText.textContent = "15 – 30 Minutes (Rapid Clearance)";
    protocolEmergencyText.textContent = "Standard response. No hospital trauma readiness required.";
  } else if (severity === 2) {
    protocolDispatchText.textContent = "2 Patrol units + 1 Contracted towing vehicle for roadway clearance.";
    protocolTrafficText.textContent = "Activate VMS Sign: 'ACCIDENT AHEAD - SLOW TRAFFIC / MERGE LEFT/RIGHT'.";
    protocolClearanceText.textContent = "30 – 60 Minutes (Moderate Clearance Window)";
    protocolEmergencyText.textContent = "Standard municipal EMS alert; no primary trauma center diversion.";
  } else if (severity === 3) {
    protocolDispatchText.textContent = "Multi-agency dispatch: State Highway Patrol + DOT Incident Response Truck + Heavy Wrecker + EMS.";
    protocolTrafficText.textContent = "Deploy dynamic speed reduction (-20 mph); activate regional overhead detour routing.";
    protocolClearanceText.textContent = "60 – 120 Minutes (Extended Corridor Impact)";
    protocolEmergencyText.textContent = "Elevated Alert: Alert nearest regional trauma center for prospective arrivals.";
  } else if (severity === 4) {
    protocolDispatchText.textContent = "CRITICAL INCIDENT: Multi-unit Emergency Command, Fire & HAZMAT, Medevac readiness, Full DOT diversion crew.";
    protocolTrafficText.textContent = "FULL CORRIDOR CLOSURE: Mandatory exit detour routing; activate statewide 511 travel alerts.";
    protocolClearanceText.textContent = "2 to 5+ Hours (Severe Structural/Hazard Clearance)";
    protocolEmergencyText.textContent = "HIGH ALERT: Trauma Level 1 readiness activation and coordinated triage.";
  }
}

function renderRiskFactors(inputs, severity) {
  riskFactorsTags.replaceChildren();

  const factors = [];

  // Weather factors
  if (inputs.visibility <= 2.0) {
    factors.push({ label: `Severe Low Visibility (${inputs.visibility} mi)`, level: "risk-high" });
  } else if (inputs.visibility <= 5.0) {
    factors.push({ label: `Reduced Visibility (${inputs.visibility} mi)`, level: "risk-moderate" });
  }

  if (inputs.precipitation > 0.1) {
    factors.push({ label: `Precipitation (${inputs.precipitation} in) - Wet Pavement`, level: "risk-high" });
  }

  if (inputs.temperature <= 32) {
    factors.push({ label: `Freezing Temperature (${inputs.temperature}°F) - Black Ice Risk`, level: "risk-high" });
  }

  if (inputs.wind_speed >= 25) {
    factors.push({ label: `High Wind Gusts (${inputs.wind_speed} mph)`, level: "risk-moderate" });
  }

  // Distance / corridor extent
  if (inputs.distance >= 4.0) {
    factors.push({ label: `Extensive Corridor Extent (${inputs.distance} mi)`, level: "risk-high" });
  } else if (inputs.distance >= 1.5) {
    factors.push({ label: `Multi-Mile Corridor Backup (${inputs.distance} mi)`, level: "risk-moderate" });
  } else {
    factors.push({ label: `Compact Incident Footprint (${inputs.distance} mi)`, level: "risk-low" });
  }

  // Lighting
  if (inputs.sunrise_sunset === "Night") {
    factors.push({ label: "Nighttime Darkness Driving Conditions", level: "risk-moderate" });
  } else {
    factors.push({ label: "Daylight Visibility Conditions", level: "risk-low" });
  }

  // Weather Condition tag
  if (["Snow", "Heavy Snow", "Heavy Rain", "Fog", "Thunderstorm"].includes(inputs.weather_condition)) {
    factors.push({ label: `Adverse Weather: ${inputs.weather_condition}`, level: "risk-high" });
  } else {
    factors.push({ label: `Weather: ${inputs.weather_condition || "Fair"}`, level: "risk-low" });
  }

  factors.forEach(f => {
    const chip = document.createElement("span");
    chip.className = `risk-chip ${f.level}`;
    chip.textContent = f.label;
    riskFactorsTags.appendChild(chip);
  });
}

// ==========================================================================
// CLIPBOARD & DISPATCH REPORT EXPORT
// ==========================================================================

function copyDispatchReportToClipboard() {
  if (!currentPredictionData || !currentInputData) {
    showToast("No active prediction to copy", "error");
    return;
  }

  const p = currentPredictionData;
  const i = currentInputData;
  const probs = p.probabilities || {};

  const reportText = `=====================================================
ROADSENSE AI - INCIDENT DISPATCH TICKET
=====================================================
PREDICTED SEVERITY: LEVEL ${p.severity} (${p.description})
MODEL CONFIDENCE: ${((probs[String(p.severity)] || 0) * 100).toFixed(1)}%

--- INCIDENT LOCATION ---
Highway/Street: ${i.street}
City / County:  ${i.city}, ${i.county} (${i.state} ${i.zipcode})
Coordinates:    ${i.start_lat}°N, ${i.start_lng}°W
Impact Distance:${i.distance} miles

--- TEMPORAL & ENVIRONMENT ---
Occurrence:     ${i.start_time} (${i.timezone || "Local"})
Lighting:       ${i.sunrise_sunset || "Day"}
Weather:        ${i.weather_condition || "Clear"} | ${i.temperature}°F | Vis: ${i.visibility} mi
Wind:           ${i.wind_speed || 0} mph ${i.wind_direction || ""}

--- OPERATIONAL PROTOCOLS ---
Dispatch:       ${protocolDispatchText.textContent}
Traffic Action: ${protocolTrafficText.textContent}
Est. Clearance: ${protocolClearanceText.textContent}
Emergency Tier: ${protocolEmergencyText.textContent}

Probability Breakdown:
  - Level 1 (Minor):       ${((probs["1"] || 0) * 100).toFixed(2)}%
  - Level 2 (Moderate):    ${((probs["2"] || 0) * 100).toFixed(2)}%
  - Level 3 (Significant): ${((probs["3"] || 0) * 100).toFixed(2)}%
  - Level 4 (Severe):      ${((probs["4"] || 0) * 100).toFixed(2)}%
=====================================================`;

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(reportText).then(() => {
      showToast("Dispatch Ticket copied to clipboard!", "success");
    }).catch(() => {
      fallbackCopyToClipboard(reportText);
    });
  } else {
    fallbackCopyToClipboard(reportText);
  }
}

function fallbackCopyToClipboard(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
  showToast("Dispatch Ticket copied to clipboard!", "success");
}

// ==========================================================================
// TOAST NOTIFICATIONS HUB
// ==========================================================================

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;

  const iconMap = {
    success: "✓",
    error: "✗",
    info: "✦"
  };

  toast.innerHTML = `<span>${iconMap[type] || "•"}</span> <span>${message}</span>`;
  toastHub.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    toast.style.transition = "opacity 0.25s, transform 0.25s";
    setTimeout(() => {
      if (toast.parentElement) toastHub.removeChild(toast);
    }, 250);
  }, 3200);
}

// ==========================================================================
// HELPER UTILITIES
// ==========================================================================

function getVal(id) {
  const el = document.getElementById(id);
  return el ? el.value.trim() : "";
}

function setVal(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

function getOptionalNumber(id) {
  const val = getVal(id);
  return val === "" ? null : parseFloat(val);
}

function getOptionalText(id) {
  const val = getVal(id);
  return val === "" ? null : val;
}