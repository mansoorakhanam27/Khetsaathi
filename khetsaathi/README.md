# KhetSathi prototype

A local-first internship MVP for tomato crop-risk awareness and advisor follow-up.

## What is implemented

- **ML:** a small logistic-regression classifier trained at startup on simulated examples. Its output is a demo risk score only; it is not trained or validated on farm data.
- **AI:** local retrieval over a small, safety-reviewed prototype guide, with grounded offline explanation generation. An optional OpenAI Responses API adapter can generate the wording when the user explicitly enables hosted AI and `OPENAI_API_KEY` is configured.
- **Agentic AI:** an orchestration flow validates inputs, calls the risk model and guidance retrieval, prepares an advisor follow-up, and waits for a human confirmation. The confirmation is stored only for the local demo session; nothing is sent.

## Run locally

Requires Python 3.10+ and no third-party Python packages.

```powershell
cd khetsaathi
python server.py
```

Open `http://127.0.0.1:8765`.

Optional hosted explanation:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "gpt-6-astra"
python server.py
```

Hosted mode sends the entered crop/weather values and retrieved guide snippets to the OpenAI Responses API only after the user checks the hosted-AI consent box. The request sets `store: false`. Do not enter personal or sensitive information. The app does not save request bodies.

## Demo case

The form starts with a sample case (fruiting tomato crop, humid conditions, rain, and leaf symptoms). Click **Run assessment** to inspect the score, retrieved advice, tool trace, and human-confirmed follow-up.

## Limits and next work

The model's labels and examples are synthetic and generated from a simple demonstration rule. This is an engineering demo, not agronomic guidance, a diagnosis, or an estimate of disease prevalence. Replace the simulated dataset with consented, agronomist-labelled local observations; evaluate by farm/time holdout; review guidance with local agricultural experts; and conduct usability and safety testing before any field use. Irrigation prompts are deliberately non-prescriptive.

## Project materials

