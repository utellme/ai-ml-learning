import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample data
data = {'Age': [25, 35, 45, 55],
        'Salary': [40000, 60000, 80000, 100000]}
df = pd.DataFrame(data)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform it
scaled_data = scaler.fit_transform(df[['Age', 'Salary']])

# Convert back to DataFrame for better visualization
scaled_df = pd.DataFrame(scaled_data, columns=['Age_scaled', 'Salary_scaled'])

print("Original Data:")
print(df)
print("\nScaled Data:")
print(scaled_df)

"""
Original Data:
   Age  Salary
0   25   40000
1   35   60000
2   45   80000
3   55  100000

Scaled Data:
   Age_scaled  Salary_scaled
0   -1.341641      -1.341641
1   -0.447214      -0.447214
2    0.447214       0.447214
3    1.341641       1.341641
"""