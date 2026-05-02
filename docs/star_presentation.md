# STAR Presentation Script — Telecom Churn Prediction
## Tech Challenge Fase 1 — 5-Minute Presentation

> **Format:** STAR (Situation → Task → Action → Result)  
> **Target duration:** 5 minutes  
> **Audience:** Technical and business evaluators  
> **Speaker notes:** Each section is timed. Maintain a steady pace — do not rush the Result section.

---

## 🎬 Opening (15 seconds)

*[Stand confidently. Make eye contact. Begin without filler words.]*

> "Good [morning/afternoon]. I'm going to walk you through an end-to-end machine learning project I built to solve a real business problem in the telecommunications industry: customer churn prediction. I'll cover the business context, what I built, how I built it, and the results — in five minutes."

---

## ⚡ Situation (45 seconds)

*[Speak from the business perspective first. Numbers create credibility.]*

> "Telecommunications companies face one of the most expensive problems in subscription businesses: **churn** — when customers cancel their service.
>
> The financial math is brutal. Acquiring a new customer costs roughly ten times more than retaining an existing one. When a customer churns, you don't just lose their monthly revenue — you lose their entire lifetime value.
>
> In our dataset, **26.5% of customers** had churned. At an average monthly charge of R$65, losing a customer represents a potential R$780 in lifetime revenue over twelve months.
>
> The business had no predictive system in place. Retention teams were operating reactively — contacting customers only after they'd already requested cancellation. By that point, the conversation is much harder, and the cost of winning them back is significantly higher.
>
> The question was: **can we predict who is about to leave before they leave?**"

---

## 🎯 Task (30 seconds)

*[Be crisp and specific about scope. Define the success criteria explicitly.]*

> "My task was to design, build, and deploy a complete machine learning solution for churn prediction — from raw data to a production-ready REST API.
>
> The project had both technical and business success criteria:
>
> - **Technical:** AUC-ROC ≥ 0.80, Recall ≥ 0.75, and F2-Score ≥ 0.70 on a held-out test set
> - **Business:** Positive net financial value — meaning the model had to generate more value from retained customers than it cost in wasted retention actions on false positives
>
> The solution also needed to be **production-ready**: modular code, automated tests, a documented API, and a monitoring plan. Not just a notebook — a system."

---

## 🔨 Action (2 minutes 30 seconds)

*[This is the technical core. Be specific about decisions and the rationale behind them.]*

> "I structured the project across four phases.
>
> **Phase 1 — Data Understanding.**
> I ran a thorough exploratory analysis on 7,043 customer records. Key findings: the dataset was moderately imbalanced at 26.5% churn. `TotalCharges` had 11 missing values requiring median imputation. The top churn signals were **tenure length**, **monthly charges**, **contract type**, and **internet service type**. Fiber optic customers churned at nearly double the rate of DSL customers.
>
> **Phase 2 — Modeling.**
> I established two baselines: a DummyClassifier as the floor, and Logistic Regression as the linear baseline. I then built the primary model — a **Multi-Layer Perceptron (MLP) in PyTorch** — with three hidden layers [128, 64, 32], BatchNorm for stable training, ReLU activations, and dropout regularization. The training loop implements **early stopping** with patience=10 and a ReduceLROnPlateau scheduler to prevent overfitting.
>
> A critical modeling decision was using `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance — this prevented the model from simply predicting 'no churn' for everyone. I also chose **F2-Score** as the primary metric, which weights Recall twice as heavily as Precision, because in our cost model, a missed churner (false negative) costs R$500 while a wasted retention action (false positive) costs only R$50 — a 10x asymmetry.
>
> Feature selection was driven by a Random Forest with SHAP analysis, reducing 36 engineered features down to 19 high-signal features — including derived features like `charges_x_monthly`, `charges_per_month`, and `service_count` — which improved both model performance and inference latency.
>
> All experiments — baselines and MLP runs — were tracked in **MLflow** with full parameter, metric, and artifact logging for reproducibility.
>
> **Phase 3 — Engineering.**
> I refactored everything from notebooks into a clean modular architecture under `src/`. The codebase separates concerns clearly: data loading, preprocessing, model training, prediction service, and HTTP routing are all independent modules. I built a **FastAPI** application with five endpoints — health check, model info, feature listing, single prediction, and batch prediction. The MLP is loaded once at startup into memory as a singleton, making per-request latency sub-20ms for the inference layer. The full preprocessing pipeline — feature engineering, one-hot encoding, median imputation, and StandardScaler — is replicated exactly at inference time.
>
> The codebase has **100 automated tests with 97% coverage**, linting via ruff with zero errors, and a Makefile for standardized commands.
>
> **Phase 4 — Documentation and MLOps.**
> I wrote a full Model Card documenting the model's purpose, limitations, biases, and monitoring requirements. I documented the deployment architecture, the preprocessing flow, and scaling strategy. I defined a five-layer monitoring plan covering data drift via Population Stability Index, concept drift via AUC-ROC on labeled cohorts, latency SLOs, error rates, and business KPIs."

---

## 📊 Result (1 minute)

*[Lead with the numbers. Then connect back to the business case.]*

> "The results exceeded all defined success criteria.
>
> On the test set — data the model never saw during training:
>
> | Metric | Target | Achieved |
> |--------|--------|----------|
> | AUC-ROC | ≥ 0.80 | **0.836** ✅ |
> | Recall | ≥ 0.75 | **0.781** ✅ |
> | F2-Score | ≥ 0.70 | **0.704** ✅ |
>
> More importantly, the business case is validated. Applying the financial model to the test set:
>
> - **DummyClassifier** (random baseline): **−R$ 102,300** — a clear loss
> - **MLP PyTorch (our model):** **+R$ 40,316** — a R$ 142,616 swing in a single test cohort
>
> That's R$28.61 of net value generated per customer scored.
>
> The system is production-ready: a fully documented REST API serving real-time predictions, a comprehensive monitoring plan with defined drift alerts and retraining triggers, and clean modular code that a new engineer could extend or maintain without the original author.
>
> If deployed to the operator's full customer base of 7,000+ customers, this model has the potential to **reduce quarterly churn by at least 15%** — the original business target — while generating a positive and measurable ROI from day one.
>
> Thank you."

---

## 💬 Anticipated Q&A (Reference Notes)

*[Not part of the 5-minute script — use as preparation]*

**Q: Why MLP and not Logistic Regression?**
> Logistic Regression achieved AUC-ROC of 0.841 — competitive, but it cannot capture non-linear interactions between features like `charges_x_monthly` and `contract_type`. The MLP with BatchNorm and early stopping generalizes better on complex feature interactions while still being deployable via a standard REST API with sub-20ms latency.

**Q: How do you handle model degradation over time?**
> The monitoring plan defines weekly PSI checks on the top 5 features and monthly AUC-ROC evaluation on labeled cohorts with a 30-60 day lag. Retraining triggers fire automatically when AUC-ROC drops below 0.80 or PSI exceeds 0.25.

**Q: What's the decision threshold?**
> Default threshold is 0.5. The experimental notebook also explored a business-optimized threshold that maximizes net value (0.44), and a P80 threshold targeting the top 20% highest-risk customers to match the retention team's operational capacity of ~500 contacts per week. These threshold strategies are ready to be exposed via API configuration.

**Q: How do you ensure LGPD compliance?**
> The model uses only aggregated behavioral and contractual data. No PII fields (name, CPF, address) are included in the feature set. Data processing for retention targeting must be backed by valid customer consent. The Model Card documents the fairness considerations and recommends quarterly bias audits.

---

*Presentation prepared by the Tech Challenge Fase 1 Group — 2026.*