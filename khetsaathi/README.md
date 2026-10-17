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

## Deploy a demo on Render

1. Push this repository to GitHub and create a **Web Service** on Render connected to the repository.
2. Set **Root Directory** to `khetsaathi`.
3. Set **Build Command** to `echo "No dependencies to install"` and **Start Command** to `python server.py`.
4. Add the environment variable `KHETSAATHI_HOST` with value `0.0.0.0`. The app reads Render's `PORT` variable automatically.
5. Deploy, open the generated `onrender.com` URL, and try the sample case.

Hosted AI is optional and stays off unless a user opts in. Do not add an API key unless you understand that the public service could use that key when users explicitly enable hosted AI. Free web services may spin down while idle, so the first visit can take longer.

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

- [Lean Canvas](../output/pdf/KhetSathi_Lean_Canvas.pdf)
- [Concept Note](../output/pdf/KhetSathi_Concept_Note.pdf)
- [Project Presentation](../output/pptx/KhetSathi_Project_Presentation.pptx)
- [Model Card](MODEL_CARD.md)
- [API Reference](API_REFERENCE.md)
