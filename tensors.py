import pandas as pd

# Create a small table (DataFrame)
data = {
    "Name": ["Amit", "Neha", "Raj"],
    "Age": [25, 30, 22],
    "City": ["Delhi", "Mumbai", "Bangalore"]
}

df = pd.DataFrame(data)

print(df)
