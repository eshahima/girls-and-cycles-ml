```markdown
# Girls & Cycles ML model: Uncertainty-Aware Menstrual Cycle Prediction

Girls & Cycles is a machine learning framework designed to replace static point predictions with calibrated confidence intervals for menstrual cycle tracking. By pairing a Bayesian prediction engine with literature-grounded priors, the model addresses cycle variability, especially for the ~20% of users with irregular cycles, while educating users on the underlying hormonal causes of their symptoms.

---

## Problem

Commercial menstrual tracking applications typically rely on point predictions, providing a single target date (e.g., "Cycle length: 28 days"). Giving a single number without a confidence interval is clinically misleading, particularly for users with irregular cycles. 

Furthermore, existing tools track symptoms in isolation without helping users understand *why* symptoms occur. Many key physiological and psychosocial symptoms, such as stress-induced dysmenorrhea or phase-dependent anxiety, are unaddressed or presented without context, leaving users without actionable educational insights into their own bodies.

---

## Approach & Methodology

To address sparse, IRB-restricted clinical data and point-prediction limitations, Girls & Cycles leverages a three-part machine learning architecture:

1. **Literature-Grounded Synthetic Dataset Generation:** Curated synthetic time-series data grounded directly in published epidemiological parameters (Fitriani et al. 2019, Patel 2023, Bull et al. 2019).
2. **Irregularity Classification:** A logistic regression classifier parameterized by published odds ratios (incorporating non-linear U-shaped BMI variance, sleep quality, and perceived stress factors).
3. **Bayesian Prediction Engine:** A probabilistic model that generates a mean cycle length prediction alongside a calibrated confidence interval ($\text{e.g., } 29.5 \pm 4.2 \text{ days}$).

---
## Dataset Validation & Literature Grounding

Because real-world menstrual cycle datasets are highly restricted due to IRB and privacy constraints, *Girls & Cycles* uses a curated synthetic dataset ($N=6,000$ cycles across 500 users) generated from published epidemiological distributions. <img width="1189" height="413" alt="image" src="https://github.com/user-attachments/assets/9ed1521f-f054-4686-bff5-515183c33c99" />

## Parameter Grounding & Literature References

### Key Parameter Summary Table

| Parameter / Feature | Value / Distribution (Synthetic Data) | Source Literature | Code Module |
| :--- | :--- | :--- | :--- |
| **Population Baseline Cycle Length** | $\mathcal{N}(30.17, 6.88^2)$ | Bull et al. (2019) | `src/data/synthetic.py` |
| **Underweight BMI Irregularity Rate** | $31.8\%$ ($\text{BMI} < 18.5$) | Patel (2023) / Fitriani (2019) | `src/models/classifier.py` |
| **Normal BMI Irregularity Rate** | $18.4\%$ ($18.5 \le \text{BMI} \le 24.9$) | Patel (2023) / Fitriani (2019) | `src/models/classifier.py` |
| **Overweight BMI Irregularity Rate** | $26.6\%$ ($\text{BMI} > 25.0$) | Patel (2023) / Fitriani (2019) | `src/models/classifier.py` |
| **Dysmenorrhea Rate (Low BMI)** | $90.3\%$ ($\text{BMI} < 18.5$) | Patel (2023) | `src/models/priors.py` |
| **Dysmenorrhea Rate (Normal BMI)** | $30.7\%$ ($18.5 \le \text{BMI} \le 24.9$) | Patel (2023) | `src/models/priors.py` |

### Primary References

* **Bull, J. R., et al. (2019).** Real-world menstrual cycle characteristics of more than 600,000 menstrual cycles. *npj Digital Medicine*, 2(1), 83.
* **Fitriani, R. J., et al. (2019).** Body mass index, sleep quality, stress conditions determine menstrual cycles among female adolescents. *International Journal of Public Health Science*, 8(1), 101–105.
* **Patel, E. M. (2023).** BMI & Menstrual Irregularities among Adolescent Girls. *Glorious International Journal of Nursing Research*, 1(1), 1–5.
* **Avila-Varela, D. S., et al. (2024).** Whole-brain dynamics across the menstrual cycle: the role of hormonal fluctuations and age in healthy women. *npj Women's Health*, 2(1), 8.
* **Iqbal, S., et al. (2021).** Menstrual cycle relation with anxiety and other psychological symptoms in women. *Indo American Journal of Pharmaceutical Sciences*, 8(2), 120–130.

---

## Future Direction & Research Goals

* **Sequential Personalized Updating:** Implement online Bayesian updating so that as an individual user logs successive cycles, the posterior distribution updates, progressively narrowing their personal prediction interval.
* **Comprehensive Symptom Contextualization:** Expand feature coverage to contextualize secondary symptoms (e.g., mood alterations, sleep disruption, dysmenorrhea) using underlying physiological drivers.
* **Faculty Collaboration:** Refine uncertainty quantification techniques for sparse clinical time-series with faculty guidance toward a peer-reviewed publication.

---

## Project Status

**In Progress** — Literature-grounded synthetic dataset curation and preliminary data validation almost complete ($N=6,000$ cycles). Bayesian prediction architecture and irregularity classification module currently in active development.

```
