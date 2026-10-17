from __future__ import annotations

import json
import math
import os
import random
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
HOST = os.getenv("KHETSAATHI_HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", os.getenv("KHETSAATHI_PORT", "8765")))

# All training examples are simulated. This makes the prototype reproducible,
# but the resulting score is not evidence of field accuracy.
FEATURES = ["temp", "humidity", "rain", "wetness", "spots", "yellowing", "fruiting"]


def sigmoid(z: float) -> float:
    if z > 35:
        return 1.0
    if z < -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def synthetic_training_rows(n: int = 2400, seed: int = 23):
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n):
        temp = rng.uniform(12, 38)
        humidity = rng.uniform(35, 100)
        rain = max(0.0, rng.gauss(4, 7))
        wetness = rng.uniform(0, 16)
        spots = int(rng.random() < 0.22)
        yellowing = int(rng.random() < 0.27)
        fruiting = int(rng.random() < 0.38)
        pressure = (
            (humidity - 65) / 13
            + (wetness - 6) / 4.2
            + min(rain, 20) / 8
            - abs(temp - 25) / 8
            + spots * 1.0
            + yellowing * 0.18
            + fruiting * 0.12
        )
        label = int(pressure > 1.0)
        if rng.random() < 0.08:
            label = 1 - label
        rows.append([temp, humidity, rain, wetness, spots, yellowing, fruiting])
        labels.append(label)
    return rows, labels


def fit_scaler(rows):
    means, stds = [], []
    for j in range(len(FEATURES)):
        values = [row[j] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        means.append(mean)
        stds.append(max(1e-6, math.sqrt(variance)))
    return means, stds


class DemoLogisticModel:
    def __init__(self):
        rows, labels = synthetic_training_rows()
        self.means, self.stds = fit_scaler(rows)
        xs = [[(x - self.means[j]) / self.stds[j] for j, x in enumerate(row)] for row in rows]
        weights = [0.0] * len(FEATURES)
        bias = 0.0
        lr, l2 = 0.035, 0.004
        for _ in range(900):
            grad = [0.0] * len(FEATURES)
            grad_b = 0.0
            for x, y in zip(xs, labels):
                p = sigmoid(sum(w * v for w, v in zip(weights, x)) + bias)
                error = p - y
                grad_b += error
                for j, value in enumerate(x):
                    grad[j] += error * value
            m = len(xs)
            bias -= lr * grad_b / m
            weights = [w - lr * (grad[j] / m + l2 * w) for j, w in enumerate(weights)]
        self.weights, self.bias = weights, bias
        self.rows, self.labels = len(rows), labels

    def predict(self, item):
        values = [item[name] for name in FEATURES]
        normalized = [(x - self.means[j]) / self.stds[j] for j, x in enumerate(values)]
        probability = sigmoid(sum(w * x for w, x in zip(self.weights, normalized)) + self.bias)
        return round(probability * 100)


MODEL = DemoLogisticModel()

GUIDANCE = [
    {
        "id": "observe",
        "title": "Record what you see",
        "tags": ["spots", "yellowing", "symptom", "risk", "high", "moderate"],
        "body": "Note the crop stage, affected leaves, and when symptoms first appeared. A risk score is a screening signal, not a diagnosis.",
        "source": "KhetSathi prototype safety guide",
    },
    {
        "id": "advisor",
        "title": "Ask a local agricultural advisor",
        "tags": ["spots", "yellowing", "symptom", "risk", "high", "moderate"],
        "body": "Share the observations and a clear photo with an extension worker or agronomist before choosing a treatment.",
        "source": "KhetSathi prototype safety guide",
    },
    {
        "id": "water",
        "title": "Check the field before irrigating",
        "tags": ["rain", "soil", "irrigation", "water", "moisture"],
        "body": "Check current soil moisture and recent rainfall before acting. This demo does not calculate a water dose or replace local crop guidance.",
        "source": "KhetSathi prototype safety guide",
    },
    {
        "id": "followup",
        "title": "Set a follow-up",
        "tags": ["follow", "reminder", "advisor", "tomorrow"],
        "body": "Recheck the plant after your chosen interval or contact an advisor sooner if symptoms spread.",
        "source": "KhetSathi prototype safety guide",
    },
]


def retrieve_guidance(item, risk):
    query = set()
    if item["spots"]:
        query.update(["spots", "symptom"])
    if item["yellowing"]:
        query.update(["yellowing", "symptom"])
    if risk >= 35:
        query.update(["risk", "moderate"])
    if risk >= 65:
        query.update(["risk", "high"])
    if item["rain"] >= 2 or item["soil_moisture"] >= 70:
        query.update(["rain", "soil", "irrigation", "water", "moisture"])
    query.update(["follow", "reminder"])
    ranked = []
    for entry in GUIDANCE:
        score = len(query.intersection(entry["tags"]))
        if score:
            ranked.append((score, entry))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [entry for _, entry in ranked[:3]]


def local_explanation(item, risk, guidance):
    level = "high" if risk >= 65 else "moderate" if risk >= 35 else "lower"
    signals = []
    if item["humidity"] >= 78:
        signals.append("high humidity input")
    if item["wetness"] >= 8:
        signals.append("long leaf-wetness input")
    if item["rain"] >= 5:
        signals.append("recent rainfall input")
    if item["spots"] or item["yellowing"]:
        signals.append("reported leaf symptoms")
    signal_text = ", ".join(signals) if signals else "the entered weather and symptom values"
    next_step = guidance[0]["body"] if guidance else "Review the observations with a local advisor."
    return (
        f"The demo model estimates a {level} risk score ({risk}/100) from {signal_text}. "
        f"This is not a diagnosis and has not been validated on farm data. {next_step}"
    )


def hosted_explanation(item, risk, guidance):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, None
    model = os.getenv("OPENAI_MODEL", "gpt-6-astra").strip()
    context = {
        "crop": "tomato",
        "crop_stage": item["stage"],
        "entered_observations": {
            "temperature_c": item["temp"], "humidity_percent": item["humidity"],
            "rain_mm": item["rain"], "leaf_wetness_hours": item["wetness"],
            "leaf_spots": bool(item["spots"]), "yellowing": bool(item["yellowing"]),
        },
        "demo_model_risk_score": risk,
        "approved_context": [{"title": x["title"], "text": x["body"]} for x in guidance],
    }
    body = json.dumps({
        "model": model,
        "store": False,
        "instructions": (
            "You are a cautious farm-support explainer for a student prototype. "
            "Use only the supplied approved_context and entered observations. "
            "Explain that the score is synthetic-demo output, not a diagnosis. "
            "Do not recommend pesticides, doses, purchases, or definitive treatment. "
            "Give one safe next step and advise local agricultural review when symptoms are present. "
            "Keep the answer under 90 words."
        ),
        "input": json.dumps(context),
        "max_output_tokens": 160,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses", data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        parts = []
        for output in payload.get("output", []):
            if output.get("type") == "message":
                for content in output.get("content", []):
                    if content.get("type") == "output_text":
                        parts.append(content.get("text", ""))
        result = "\n".join(part for part in parts if part).strip()
        if result:
            return result, model
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return None, None
    return None, None


def validate(payload):
    numeric = {
        "temp": (float, -5, 55), "humidity": (int, 0, 100),
        "rain": (float, 0, 250), "wetness": (float, 0, 24),
        "soil_moisture": (int, 0, 100),
    }
    clean = {}
    for key, (cast, low, high) in numeric.items():
        try:
            value = cast(payload.get(key))
        except (TypeError, ValueError):
            raise ValueError(f"Enter a valid value for {key}.")
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
        clean[key] = value
    clean["spots"] = bool(payload.get("spots"))
    clean["yellowing"] = bool(payload.get("yellowing"))
    clean["fruiting"] = payload.get("stage") in {"flowering", "fruiting"}
    clean["stage"] = payload.get("stage", "vegetative")
    if clean["stage"] not in {"seedling", "vegetative", "flowering", "fruiting"}:
        raise ValueError("Choose a valid crop stage.")
    return clean


def analyze(payload):
    item = validate(payload)
    risk = MODEL.predict(item)
    level = "High" if risk >= 65 else "Moderate" if risk >= 35 else "Lower"
    guidance = retrieve_guidance(item, risk)
    trace = [
        {"tool": "input_check", "status": "complete", "detail": "Required field values and ranges checked."},
        {"tool": "disease_risk_model", "status": "complete", "detail": "Synthetic-demo logistic regression returned a screening score."},
        {"tool": "guidance_retrieval", "status": "complete", "detail": f"Retrieved {len(guidance)} local safety-guide snippets."},
        {"tool": "follow_up_planner", "status": "awaiting_user", "detail": "Prepared an optional advisor follow-up; no message has been sent."},
    ]
    explanation, model = None, None
    if payload.get("useHostedAI"):
        explanation, model = hosted_explanation(item, risk, guidance)
    mode = "OpenAI Responses API" if explanation else "Offline grounded explanation"
    explanation = explanation or local_explanation(item, risk, guidance)
    signals = []
    if item["humidity"] >= 78: signals.append("Humidity entered in a high range")
    if item["wetness"] >= 8: signals.append("Long leaf-wetness duration entered")
    if item["rain"] >= 5: signals.append("Recent rainfall entered")
    if item["spots"]: signals.append("Leaf spots reported")
    if item["yellowing"]: signals.append("Yellowing reported")
    if not signals: signals.append("No elevated input signal crossed the demo thresholds")
    irrigation = (
        "Check field moisture and local guidance before irrigating."
        if item["soil_moisture"] >= 70 or item["rain"] >= 5
        else "Use your normal field check and local schedule; this demo does not set a water dose."
    )
    return {
        "risk": risk, "level": level, "signals": signals,
        "explanation": explanation, "explanation_mode": mode, "model": model,
        "guidance": guidance, "trace": trace, "irrigation_prompt": irrigation,
        "follow_up": "Ask an agricultural advisor to review the observations; confirm locally before any treatment decision.",
        "synthetic_warning": "Demonstration model trained on simulated examples only. Not for real farm decisions.",
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Avoid writing field values or request bodies to logs.
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        routes = {"/": (ROOT / "index.html", "text/html; charset=utf-8"),
                  "/static/style.css": (STATIC / "style.css", "text/css; charset=utf-8"),
                  "/static/app.js": (STATIC / "app.js", "text/javascript; charset=utf-8")}
        if self.path == "/api/health":
            return self.send_json(200, {"status": "ready", "model": "synthetic logistic regression", "synthetic_rows": MODEL.rows,
                                        "hosted_ai_configured": bool(os.getenv("OPENAI_API_KEY"))})
        file_entry = routes.get(self.path)
        if not file_entry:
            self.send_error(404)
            return
        file_path, content_type = file_entry
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path not in {"/api/analyze", "/api/confirm"}:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 12000:
                return self.send_json(413, {"error": "Request is too large."})
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return self.send_json(400, {"error": "Request must contain valid JSON."})
        if self.path == "/api/confirm":
            return self.send_json(200, {"status": "confirmed locally", "message": "Follow-up recorded for this demo session only. No message was sent."})
        try:
            return self.send_json(200, analyze(payload))
        except ValueError as error:
            return self.send_json(400, {"error": str(error)})


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"KhetSathi is available at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
