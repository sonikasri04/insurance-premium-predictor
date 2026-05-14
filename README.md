# Health Insurance Premium Predictor

This project builds a production-ready insurance premium pricing model that mirrors actuarial risk modeling used in the insurtech industry. It combines exploratory data analysis, ensemble ML modeling, SHAP-based explainability, and a bias/fairness audit deployed as an interactive web application.

Live Demo: https://healthinsurance-premium-predictor.streamlit.app/

## Architecture
 
```
├── Data Ingestion & Validation
│   ├── US Health Insurance Dataset     (1,338 records, 7 features)
│   └── Enhanced Claims Dataset         (4,500 records, 17 features)
│
├── Feature Engineering
│   ├── Categorical encoding            (sex, smoker, region)
│   └── Interaction features            (bmi_smoker, age_smoker, age_bmi)
│
├── Modeling Pipeline
│   ├── Baseline — Linear Regression    R² = 0.866
│   ├── Ensemble — Random Forest        R² = 0.871
│   ├── Boosting — XGBoost              R² = 0.849
│   └── Tuned Random Forest             R² = 0.875 ✅ (selected)
│
├── Explainability
│   ├── SHAP TreeExplainer
│   ├── Beeswarm & bar summary plots
│   └── Per-prediction waterfall charts
│
├── Fairness Audit
│   ├── Prediction error by sex
│   └── Prediction error by smoker status
│
└── Deployment
    └── Streamlit Cloud (free tier)
```
 
---

## Model Performance
 
| Model | RMSE | R² | CV R² (5-fold) |
|---|---|---|---|
| Linear Regression | 4,567 | 0.866 | 0.837 |
| Random Forest | 4,481 | 0.871 | 0.837 |
| XGBoost | 4,848 | 0.849 | 0.816 |
| **Tuned Random Forest** | **4,406** | **0.875** | — |
 
> Hyperparameters: `max_depth=10`, `min_samples_split=10`, `n_estimators=200`
 
---

## Key Findings
 
### Feature Importance (SHAP)
- **`bmi_smoker`** is the single most dominant feature — the interaction between BMI and smoking status contributes more to premium variance than any individual feature alone
- **Age** is the second strongest predictor (SHAP mean = ~2,800)
- **Region** and **sex** have near-zero predictive power — consistent with regulatory fairness expectations
### Fairness Audit Results
| Group | Mean Prediction Error | Std |
|---|---|---|
| Female | -$87 | $4,372 |
| Male | -$620 | $4,432 |
| Non-smoker | -$380 | $4,365 |
| Smoker | -$187 | $4,577 |
 
> No group is systematically overcharged by a significant margin. The model demonstrates reasonable demographic fairness across sex and smoking status.
 
---

---
 
## ⚙️ Feature Engineering
 
Three interaction features were engineered to capture non-linear risk compounding:
 
```python
df['bmi_smoker'] = df['bmi'] * df['smoker']    # dominant risk driver
df['age_smoker'] = df['age'] * df['smoker']    # age-amplified smoking risk
df['age_bmi']    = df['age'] * df['bmi']       # metabolic risk over time
```
 
These alone improved Linear Regression R² from ~0.75 to 0.866 — demonstrating the value of domain-driven feature construction over raw algorithmic power.
 
---
    
