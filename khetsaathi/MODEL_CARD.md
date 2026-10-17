# Model card: KhetSathi demonstration classifier

- **Task:** classify a synthetic example as a higher/lower tomato disease-pressure scenario, then display a 0-100 screening score.
- **Model:** logistic regression implemented with Python's standard library; trained at startup with seeded synthetic rows.
- **Training examples:** 2,400 generated rows. Labels are generated from a transparent heuristic with a small amount of simulated label noise.
- **Intended use:** demonstrate an ML inference step and how an agent can consume its output in a student project.
- **Not intended for:** diagnosis, treatment selection, pesticide advice, yield prediction, water-dose decisions, or real farm use.
- **Evaluation:** no real-world metric is claimed. A future pilot must use agronomist-labelled data, farm/time-separated validation, class-wise metrics, calibration, and error review.
- **Known limits:** synthetic label construction makes the model reflect the generating heuristic; it cannot establish field performance or generalization.
