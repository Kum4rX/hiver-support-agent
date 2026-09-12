# Engineering Evaluation & Architecture Report: Hiver AppleSupport AI Agent

## Executive Summary

This report documents the design, implementation, and empirical evaluation of an end-to-end, production-oriented Customer Support AI Agent built on historical AppleSupport Twitter interactions. The system automates triage and resolution synthesis while maintaining strict safety, brand tone, platform guardrails, deterministic out-of-domain rejection, and multi-intent acknowledgment.

The core design centers on a multi-stage deterministic and neural pipeline:
1. **Preprocessing & Normalization** (Unicode normalization, tweet mention/URL stripping, token sanitation).
2. **Intent Classification** (11-class customer issue taxonomy with Keyword/Rule baseline and TF-IDF statistical classifier).
3. **Deterministic Safety & Risk Escalation Engine** (Microsecond-latency rule hierarchy for physical hazards, account compromise, fraud, and legal triggers).
4. **Out-of-Domain Guard** (Deterministic non-Apple platform/device redirection).
5. **Dense Semantic Retrieval** (FAISS `IndexFlatIP` querying 65,239 pre-filtered historical AppleSupport customer-agent pairs).
6. **Grounded Response Generation** (Deterministic resolution extractor and brand synthesis engine operating 100% offline with optional pluggable LLM interfaces).
7. **Response Guardrails** (Strict Twitter $\le 280$ character limit enforcement, PII protection, and safety override protocols).


### Evaluation Taxonomy & Integrity Disclosures (Categories A–F)

To preserve scientific integrity and prevent metric conflation, all empirical findings are categorized into six distinct evaluation tiers:

- **Tier A — Verified Human-Labelled Benchmark**: Exactly 11 verified human labels in `golden_set.csv` (10 `BATTERY_POWER`, 1 `DEVICE_PERFORMANCE`). Used as an authentic small-sample pipeline sanity check.
- **Tier B — Provisional / Auto-Labelled Exploratory Benchmark**: 189 candidate suggestions in `data/golden_evaluation_provisional.csv` (`label_source = "auto_provisional"`). Used strictly for pipeline smoke testing across all 11 intent classes; **never cited as authentic ground truth**.
- **Tier C — Curated Safety Tests**: 21 adversarial and benign test queries (13 hazards, 8 benign) evaluating rule hierarchy recall. Not a natural customer distribution benchmark.
- **Tier D — Retrieval Similarity Metrics**: Cosine similarity ($\ge 0.35$ relevance hit rate) over 65,239 pre-filtered document pairs. Measures dense vector proximity, **NOT human-annotated factual relevance**.
- **Tier E — LLM Judge Results**: 6-dimension rubric (Groundedness, Relevance, Actionability, Safety, Tone, Policy Constraints) implemented in `src/evaluation/llm_judge.py`. Reported as **"Not measured"** when unconfigured without an API key.
- **Tier F — Human Agreement Results**: Quadratic weighted Cohen's Kappa ($\kappa_w$) and Spearman rank correlation ($\rho$). Reported as **"Not measured"** because human quality rating columns in `data/human_response_quality_template.csv` are blank.

---

## 1. System Architecture


```
                       Customer Tweet
                             │
                             ▼
               ┌───────────────────────────┐
               │     1. Preprocessing      │
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │ 2. Intent Classification  │  (Includes Out-of-Domain & Multi-Intent)
               └─────────────┬─────────────┘
                             │
               ┌─────────────┴─────────────┐
               │ Is Out-of-Domain? (Dell/  │ ──► [OUT-OF-DOMAIN BOUNDARY RESPONSE]
               │ Windows/Android/Samsung)  │
               └─────────────┬─────────────┘
                             │ Safe & In-Domain
                             ▼
               ┌───────────────────────────┐
               │  3. Escalation Engine     │
               └─────────────┬─────────────┘
                             │
               ┌─────────────┴─────────────┐
               │ Is Escalated == True?     │
               │ (Hardware Danger/Sec Risk)│
               └─────────────┬─────────────┘
                             │
            ┌────────────────┴────────────────┐
     YES    │                                 │   NO (Safe)
            ▼                                 ▼
┌──────────────────────────┐      ┌──────────────────────────┐
│  Safety Action Protocol  │      │  4. FAISS Dense Retrieval│
│  (Routing + Link)        │      │  (65,239 Filtered Pairs) │
└───────────┬──────────────┘      └───────────┬──────────────┘
            │                                 │
            │                                 ▼
            │                     ┌──────────────────────────┐
            │                     │ 5. Grounded Generator    │
            │                     │ (Dual-Intent Synthesizer)│
            │                     └───────────┬──────────────┘
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │   6. Response Guardrails  │
               │ (<=280 Chars & PII Check) │
               └─────────────┬─────────────┘
                             │
                             ▼
                    Final Tweet Response
```

---

## 2. Intent Classification Component

### 2.1 Taxonomy Definition (11 Classes + Out-of-Domain)
The customer intent space is partitioned into 11 distinct operational categories:
1. `BATTERY_POWER`: Battery drain, rapid discharge, charging failure, overheating while charging.
2. `CONNECTIVITY`: Wi-Fi drops, Bluetooth pairing, cellular/LTE/5G data, hotspot, router issues.
3. `CALLS_COMMUNICATION`: Dropped phone calls, FaceTime errors, iMessage/SMS delivery, voicemail.
4. `DEVICE_PERFORMANCE`: System lag, freezing, random reboots, app crashes, iOS update glitches.
5. `KEYBOARD_INPUT`: Autocorrect bugs, typing delays, predictive text, missing keys.
6. `APPS_MEDIA`: App Store download errors, third-party apps, Apple Music, Photos, Podcasts.
7. `DISPLAY_AUDIO_CAMERA`: Screen black/flickering, touch unresponsiveness, audio/mic issues, camera blur.
8. `ACCOUNT_ICLOUD`: Apple ID lockout, password reset, 2FA verification, iCloud storage.
9. `PURCHASE_PAYMENT`: Unauthorized charges, subscription renewals, billing disputes, Apple Pay.
10. `HOW_TO_OTHER`: General feature configuration, iOS navigation, settings inquiries.
11. `SECURITY`: Suspected hacking, stolen devices, phishing attempts, unauthorized access.
12. `OUT_OF_DOMAIN`: Explicit non-Apple platforms or hardware (Windows, Dell, Android, Samsung, HP, Linux, etc.).

### 2.2 Models Implemented
- **Keyword/Rule Baseline (`KeywordRuleIntentClassifier`)**: Deterministic priority matching over compiled regex patterns and keyword sets.
- **TF-IDF + Logistic Regression (`TfidfLogisticIntentClassifier`)**: Sublinear term-frequency vectorizer with n-grams $(1, 2)$ paired with balanced multinomial logistic regression.
- **Hybrid Intent Classifier (`HybridIntentClassifier`)**: Production orchestrator that balances ML probabilities against deterministic taxonomy rules, with safety overrides for security intents and multi-intent detection.

### 2.3 Out-of-Domain Guard & Multi-Intent Support
- **Out-of-Domain Handling**: Queries mentioning non-Apple entities (e.g. Dell, Windows 11, Samsung) are caught deterministically, preventing the agent from hallucinating Apple device troubleshooting.
- **Multi-Intent Detection**: When a query presents multiple strong intent signals (e.g. battery drain + Wi-Fi disconnection), the system preserves the primary routing intent while synthesizing a combined acknowledgment reply.

### 2.4 Empirical Evaluation & Dataset Limitations

> [!IMPORTANT]
> **Data Volume Disclosure**:
> - The verified human-reviewed benchmark `golden_set.csv` currently contains **11 verified samples** (10 `BATTERY_POWER`, 1 `DEVICE_PERFORMANCE`).
> - An expanded 200-sample dataset `data/golden_evaluation_provisional.csv` was generated from candidate pool suggestions (`label_source = "auto_provisional"`).
> - **Provisional labels are NOT equivalent to hand-labelled ground truth.** They reflect concordant candidate suggestions and are exploratory sanity checks.
> - The assignment requirement of approximately **150–250 hand-labelled examples remains incomplete** until full human review is conducted.

#### Benchmark Intent Evaluation Results (Human-Labelled Ground Truth):
*The official benchmark metrics are measured exclusively on the 11 verified human-labelled samples without inflating scores:*

| Classifier Model | Sample Count | Provenance | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Keyword/Rule Baseline** | 11 | Human Verified | 0.9091 | 0.8264 | 0.9091 | 0.8658 |
| **Hybrid (Rule + TF-IDF)** | 11 | Human Verified | 0.8182 | 0.8264 | 0.8182 | 0.8182 |

#### Exploratory Provisional Evaluation Results:
*Measured on `data/golden_evaluation_provisional.csv`. Provided strictly for pipeline code verification across all 11 taxonomy classes:*

| Evaluation Subset | Sample Count | Provenance / Label Source | Hybrid Accuracy | Hybrid Weighted F1 | Interpretation Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Subset A: Human-Only** | 11 | `human` (Verified) | 0.8182 | 0.8182 | Official small-sample sanity check |
| **Subset B: Auto-Provisional** | 189 | `auto_provisional` | 1.0000 | 1.0000 | **Provisional / auto-labelled — not ground truth** |
| **Subset C: Combined Provisional** | 200 | 11 `human` + 189 `auto_prov` | 0.9900 | 0.9924 | **Provisional / auto-labelled — not ground truth** |

> [!WARNING]
> **Why Provisional Scores (F1 ~0.99) Are Not Ground Truth**:
> The provisional labels were derived from candidate pool suggestions where keyword and classifier predictions were concordant. Evaluating against these labels naturally yields near-perfect metrics (~0.99 F1), but this is circular verification of classifier consistency, **not independent human validation**. True production benchmark evaluation requires manual human labeling.

---

## 3. Deterministic Risk & Escalation Engine

Customer safety and financial security are evaluated on a strict deterministic hierarchy:
1. **Physical Safety Hazards (Severity: CRITICAL)**: Swollen/bulging battery, smoke, sparks, electrical shock, burning smell.
2. **Account Security Compromise (Severity: HIGH)**: Hacked Apple ID, unauthorized email/password change, phishing scams.
3. **Financial Fraud (Severity: HIGH)**: Unauthorized credit card charges, disputed subscriptions.
4. **Legal / Regulatory (Severity: HIGH)**: Mentions of attorney, lawsuit, police report, FTC complaint.
5. **Chronic Unresolved Frustration (Severity: MEDIUM)**: Explicit demand for supervisor, repeat failed repairs.

### 3.1 Empirical Evaluation on Curated Safety Test Suite:
- **Total Test Cases**: 21 (13 safety/risk triggers + 8 benign queries)
- **Safety Hazard Recall**: **100.00%** (13/13 hazards detected)
- **Benign Precision / Specificity**: **100.00%** (8/8 benign queries passed without false alarms)
- **False Negatives (Missed Hazards)**: **0**
- **False Positives (Over-escalation)**: **0**
- **Average Engine Latency**: **30.05 microseconds ($\mu$s)**

---

## 4. FAISS Dense Retrieval Component

The retriever connects to the pre-indexed filtered corpus of **65,239 verified AppleSupport document pairs** using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional normalized embeddings) with FAISS `IndexFlatIP`.

### 4.1 Retrieval Benchmark Metrics:
- **Corpus Size**: 65,239 historical documents.
- **Mean Top-1 Cosine Similarity**: **0.7510**
- **Mean Top-3 Average Cosine Similarity**: **0.7318**
- **Relevance Hit Rate ($\text{similarity} \ge 0.35$)**: **100.0%**
- **Mean Search Latency**: **93.73 ms** per query.

Sample Retrieval Precision:
- Query: *"My iPhone battery is draining in less than 3 hours after updating to iOS 11."*
  - Top Match: *"Why is my iPhone battery draining so rapidly after the update?..."* (Similarity: **0.8666**)
- Query: *"How do I cancel my Apple Music subscription before the free trial ends?"*
  - Top Match: *"How do I cancel Apple Music..."* (Similarity: **0.8628**)

---

## 5. Response Generation & Guardrails

### 5.1 Deterministic Grounded Generation
To guarantee reliability and 100% offline reproducibility without requiring API keys:
1. The generator extracts actionable troubleshooting steps directly from the highest-ranking retrieved AppleSupport resolution (`support_text_clean`).
2. It pairs this with intent-specific empathetic openings.
3. For multi-intent queries, it synthesizes dual-intent acknowledgment.
4. For out-of-domain queries, it issues a polite platform boundary statement.

### 5.2 Response Guardrails Validation
- **Strict Twitter Character Limit ($\le 280$ chars)**: Enforced via intelligent sentence-boundary truncation.
- **PII / Privacy Safety**: Prohibits public solicitation of passwords, CVVs, or PINs.
- **Safety Overrides**: Suppresses standard advice when escalation is triggered.

#### Measured Guardrail & Quality Results (14 Diverse Test Scenarios):
- **Twitter Length Compliance ($\le 280$ chars)**: **100.0%** (14/14)
- **PII Privacy Compliance**: **100.0%** (14/14)
- **Deterministic Actionability / Usefulness**: **57.1%** (8/14 - reflects that some Twitter support replies are diagnostic triage questions like *"Are you on iOS 11.0.3?"* rather than full multi-step tutorials).
- **LLM-as-a-Judge Quality Score**: **NOT MEASURED** (Unconfigured; requires optional `LLM_JUDGE_API_KEY`).
- **Judge-Human Agreement**: **NOT MEASURED** (Current human Golden Set contains intent/escalation labels, not independent reply-quality ratings).
- **Response Character Length**: Min: 143 chars, Mean: 204.0 chars, Max: 265 chars.
- **Mean End-to-End Pipeline Latency**: **225.37 ms** (including vector model execution).

### 5.3 LLM-as-a-Judge Evaluation

The evaluation harness implements an extensible LLM-as-a-judge module in `src/evaluation/llm_judge.py` to evaluate response quality beyond deterministic keyword heuristics.

#### Rubric Dimensions & Scoring Scale (1–5 Likert Scale):
1. **Groundedness (1–5)**: Evaluates whether the generated response is factually supported by retrieved historical AppleSupport evidence or official safety protocol, penalizing hallucinated advice or ungrounded claims.
2. **Relevance (1–5)**: Evaluates whether the reply directly addresses the specific customer problem rather than deflecting or answering an unasked question.
3. **Actionability (1–5)**: Evaluates whether the reply provides a concrete, executable troubleshooting step (e.g. specific Settings path, key combo, official link, or safety instruction).
4. **Safety (1–5)**: Validates that critical physical hazards (smoking, swelling) immediately instruct power disconnection, account compromises route to official security tools, and out-of-domain queries issue polite boundaries.
5. **Tone (1–5)**: Evaluates whether the language is professional, empathetic, concise, and aligned with AppleSupport Twitter conventions.
6. **Policy Constraints**: Enforces zero PII solicitation and strict compliance with the Twitter $\le 280$ character limit.

#### Structured Output JSON Schema:
```json
{
  "groundedness": 1-5,
  "relevance": 1-5,
  "actionability": 1-5,
  "safety": 1-5,
  "tone": 1-5,
  "overall": 1-5,
  "reason": "<concise explanation in 1-2 sentences>"
}
```

#### Evaluation Sample Selection:
The judge suite operates over a stratified set of 14 pipeline scenarios spanning:
- Standard in-domain technical inquiries (Battery drain, Wi-Fi errors, Autocorrect glitch, Black screen, Subscription cancellation)
- Critical hardware safety hazards (Melting/smoking charger)
- Account security breaches (Compromised Apple ID lockout)
- Out-of-domain queries (Windows 11 Dell laptop, Samsung Galaxy)
- Multi-intent queries (Battery drain + Wi-Fi drops)
- Boundary / under-specified queries ("help")

#### Execution Status & Empirical Measurement:
- **LLM Judge Execution**: **Not measured** (no external `LLM_JUDGE_API_KEY` was configured in this evaluation environment). In accordance with scientific integrity guidelines, no synthetic or fabricated scores are generated.
- **Judge-Human Agreement**: **Not measured** — the current 11-example human Golden Set contains intent/escalation labels, not independent human reply-quality ratings, so judge-human agreement cannot currently be claimed.
- **Future Human Quality Annotation**: A clean rating template with pipeline responses generated for the verified human set has been provided at `data/human_response_quality_template.csv` with blank scoring columns ready for independent human annotation. Agreement calculation via quadratic weighted Cohen's Kappa ($\kappa_w$) and Spearman rank correlation ($\rho$) is pre-implemented in `src/evaluation/llm_judge.py`.

---

## 6. Failure Analysis & Edge-Case Diagnoses

| Scenario Category | Example Query | Observed Behavior | Root Cause & Mitigation |
| :--- | :--- | :--- | :--- |
| **Multi-Intent Ambiguity** | *"Battery dies in 2 hours and WiFi won't connect."* | Primary: `CONNECTIVITY`, Secondary: `BATTERY_POWER` | Dual-intent detected. **Mitigation**: Synthesizes combined acknowledgment ("We can help with both your Wi-Fi and battery life...") within 265 chars. |
| **Out-of-Domain Query** | *"Help me fix blue screen on Windows 11 Dell laptop."* | Intent: `OUT_OF_DOMAIN` (Windows 11) | Non-Apple device query. **Mitigation**: Out-of-Domain Guard intercepts query in 0.24 ms and returns polite platform boundary statement. |
| **Slang & Typos** | *"yo my fon iz glitchin super bad nd battry dyin af"* | Intent: `DEVICE_PERFORMANCE` | Word forms deviate from standard spelling. **Mitigation**: Sub-word dense embeddings handle semantic intent mapping. |
| **Subtle Heat vs Hazard** | *"My iPhone feels a bit warm when playing games."* | Escalation: `False` (Safe) | Benign operating temperature. **Mitigation**: Boundary regex strictly requires hazard tokens (*smoke, swelling, burning*) before escalating. |
| **Explicit Physical Hazard** | *"Battery is swelling and pushing the screen up, I smell burning."* | Escalation: `True` (`PHYSICAL_SAFETY_HAZARD`) | Immediate critical safety risk. **Mitigation**: Overrides standard generation, instructs power disconnect, routes to Safety Team in 0.55 ms. |
| **Ultra-Short Query** | *"It's broken please fix."* | Triage prompt returned | Zero diagnostic detail. **Mitigation**: Validator catches under-specified input and asks for clarifying details. |

---

## 7. What is Misleading About My Headline Numbers?

In the spirit of scientific integrity and engineering transparency, we explicitly document where headline metrics must be interpreted with caution:

1. **The 0.8658 Intent F1 Headline Number is NOT a Production Metric**:
   - The verified human-reviewed `golden_set.csv` contains only **11 verified samples** (10 `BATTERY_POWER`, 1 `DEVICE_PERFORMANCE`).
   - An expanded 200-sample dataset `data/golden_evaluation_provisional.csv` contains these 11 verified samples plus 189 auto-provisional candidate suggestions (`label_source = "auto_provisional"`).
   - Evaluating on the 200-sample provisional dataset yields ~0.9924 F1, but this score is exploratory and circular because provisional labels were filtered for concordance with candidate suggestions. It must **never** be presented as an authentic human benchmark.
   - The assignment requirement of approximately 150–250 hand-labelled examples remains incomplete until full human review has occurred.
   - Reporting 0.8658 on 11 human samples is a small-sample sanity check of the code execution path, **not statistical evidence of model generalization**. A production benchmark requires a balanced 150–250 hand-reviewed golden set.

2. **FAISS Cosine Similarity ($\ge 0.35$) is NOT Human Relevance**:
   - A 100.0% retrieval hit rate indicates that the dense retriever found vector neighbors with cosine similarity above threshold in the 65,239-document corpus.
   - Dense embeddings capture semantic proximity, but semantic similarity does not guarantee factual resolution correctness. True retrieval precision requires human relevance annotation.

3. **Escalation 100% Recall is on a Curated Benchmark**:
   - The 100% safety recall was measured against 21 curated adversarial test cases covering known hardware hazards, security theft, and billing disputes.
   - In production, novel user phrasings, multilingual tweets, or unusual metaphors may evade regular expressions unless periodically updated.

4. **Response Quality has NOT Been LLM-Judged (Judge-Human Agreement is Unmeasured)**:
   - Our response evaluation verifies character limits ($\le 280$), PII safety, safety routing, and presence of troubleshooting vocabulary deterministically.
   - An LLM-as-a-judge rubric (Groundedness, Relevance, Actionability, Safety, Tone on a 1–5 scale) is implemented in `src/evaluation/llm_judge.py` but was **not executed** because no external API key was configured.
   - Crucially, **the current 11-example human Golden Set contains intent/escalation labels, not independent human reply-quality ratings, so judge-human agreement cannot currently be claimed.**
   - We explicitly state "Not measured" rather than inventing synthetic agreement percentages.

5. **Historical Twitter Data is Frequently DM-Oriented**:
   - Real-world AppleSupport tweets often ask the user for context or invite them to DM due to privacy and public character constraints.
   - While our pre-filtering stripped pure boilerplate (e.g. *"please DM us"*), some retrieved solutions still reflect first-line triage questions rather than exhaustive tutorials.

---

## 8. Summary of Final Measured Metrics

```
================================================================================
                          MASTER EVALUATION REPORT CARD
================================================================================
Component                        | Primary Metric           | Measured Value    
--------------------------------------------------------------------------------
Intent (Rule Baseline)           | Weighted F1 Score        | 0.8658 (11 samples - Prov.)
Intent (Hybrid Model)            | Weighted F1 Score        | 0.8182 (11 samples - Prov.)
Escalation Safety Engine         | Hazard Safety Recall     | 100.0% (13/13)
Escalation Safety Engine         | Benign Specificity (TNR) | 100.0% (8/8)
Escalation Latency               | Average Latency          | 30.05 µs
FAISS Retrieval (65k docs)       | Mean Top-1 Cosine Sim    | 0.7510
FAISS Retrieval (65k docs)       | Relevance Hit Rate       | 100.0%
FAISS Retrieval Latency          | Mean Search Time         | 93.73 ms
Twitter Length Guardrail         | Compliance (<=280 chars) | 100.0%
PII Privacy Guardrail            | Compliance Rate          | 100.0%
Actionable Quality Check         | Deterministic Pass Rate  | 57.1%
LLM-as-a-Judge Quality           | 6-Dimension Rubric Score | NOT MEASURED (Optional)
Judge-Human Agreement            | Weighted Cohen's Kappa   | NOT MEASURED (No human ratings)
Response Character Length        | Average Length           | 204.0 chars
End-to-End Pipeline Latency      | Mean Latency             | 225.37 ms
================================================================================
```

---

## 9. Limitations & Future Work

1. **Golden Set Scope & Provenance**:
   > The required 150–250 hand-labelled Golden Set was not completed in this iteration. The verified human-reviewed benchmark contains 11 samples. While an exploratory 200-sample dataset (`data/golden_evaluation_provisional.csv`) is provided for pipeline verification, its auto-provisional metrics are exploratory and must not be interpreted as definitive production benchmarks.
   
   The human-reviewed benchmark currently contains 11 verified samples used as a functional sanity check. The evaluation harness, comparative subset runner (`eval_intent.py --all-subsets`), and `golden_reviewer.py` workflow are fully implemented and ready to ingest a full 200-sample hand-labelled golden set without any architectural modifications. Full human verification of the candidate pool remains the primary prerequisite before declaring production readiness.

2. **Retrieval Semantic vs. Factual Accuracy**:
   FAISS vector retrieval demonstrates strong cosine similarity (mean 0.7510 across 65,239 documents), but cosine similarity measures vector alignment rather than human-verified factual accuracy. A future iteration will integrate human relevance judgments for Top-1 and Top-3 matches.

3. **External LLM Integration**:
   The response generator runs 100% offline and deterministic to ensure reproducibility, low latency (14.69 ms), and zero cost. For production deployments with rich multi-paragraph inquiries, an optional LLM synthesizer (e.g. Claude 3.5 Sonnet or Gemini 1.5 Pro) with strict length guardrails can be enabled.

4. **Human Annotation & Agreement Procedures**:
   - **Accelerated Golden Set Review**: Reviewers execute `python golden_reviewer.py --reviewer <name>`. The CLI presents the remaining 189 candidates from `data/golden_evaluation_provisional.csv` with a non-ground-truth warning banner. Reviewers can accept (`[Enter]`), change (`[1-11]`), escalate (`[e]`), or skip (`[s]`). All decisions write immediately to `golden_set.csv` (`label_source="human"`) and append audit metadata to `data/golden_annotation_audit.jsonl`.
   - **Human Reply Quality Rating**: Reviewers fill in the blank 1–5 scoring columns in `data/human_response_quality_template.csv` across Groundedness, Relevance, Actionability, Safety, Tone, and Overall Quality.
   - **Agreement Computation**: Execute `python src/evaluation/llm_judge.py --calculate-agreement` to compute quadratic weighted Cohen's Kappa and Spearman correlation once human ratings are present. When unrated, the harness truthfully reports "Not measured".


