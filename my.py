import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt

# 1. Generate synthetic HCT dataset
np.random.seed(42)
n = 100
data = pd.DataFrame({
    'age': np.random.randint(20, 70, n),
    'disease_risk': np.random.choice([0, 1], n),  # 0: low, 1: high
    'donor_type': np.random.choice([0, 1], n),    # 0: related, 1: unrelated
    'time': np.random.exponential(24, n),         # months until event/censor
    'event': np.random.binomial(1, 0.7, n)        # 1: death, 0: censored
})

# 2. Fit Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(data, duration_col='time', event_col='event')
print("Cox Model Summary:")
print(cph.summary)

# 3. Predict survival for a new patient
new_patient = pd.DataFrame({
    'age': [45],
    'disease_risk': [1],
    'donor_type': [0]
})
survival_func = cph.predict_survival_function(new_patient)
survival_func.plot()
plt.title("Predicted Survival Curve for New Patient")
plt.xlabel("Months after HCT")
plt.ylabel("Survival Probability")
plt.show()

# 4. Interpretation
print("\nInterpretation:")
print("Coefficients > 0 mean higher risk (lower survival).")
print("For example, if 'disease_risk' has a positive coefficient, high-risk disease increases hazard (worse prognosis).")
print("The survival curve shows the probability of survival over time for the new patient profile.")
# ADDED new comment
# ADDED new comment 2

# Optional: Kaplan-Meier for overall survival
kmf = KaplanMeierFitter()
kmf.fit(data['time'], event_observed=data['event'])
kmf.plot_survival_function()
plt.title("Overall Kaplan-Meier Survival Curve")
plt.xlabel("Months after HCT")
plt.ylabel("Survival Probability")
plt.show()
