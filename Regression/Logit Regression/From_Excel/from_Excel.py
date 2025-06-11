import pandas as pd
import statsmodels.api as sm

# Load Excel data
file_path = 'logit_data.xlsx'  # Update with your file path
df = pd.read_excel(file_path)

# Define independent variables (predictors)
X = df[['age', 'income']]  # Use more predictors if available
X = sm.add_constant(X)     # Adds intercept term

# Define dependent variable (binary outcome)
y = df['clicked_ad']

# Fit logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Output summary
print(result.summary())