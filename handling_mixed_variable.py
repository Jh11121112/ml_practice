import numpy as np
import pandas as pd

df = pd.read_csv('/Users/janhvihaldwania/ml_practice/data_folder/titanic.csv')

df['number'].unique()
# extract numerical part
df['number_numerical'] = pd.to_numeric(df["number"],errors='coerce',downcast='integer')

# extract categorical part
df['number_categorical'] = np.where(df['number_numerical'].isnull(),df['number'],np.nan)

df.head()

# Extract numeric part from Cabin column
# (\d+) means extract digits/numbers
df['cabin_num'] = df['Cabin'].str.extract(r'(\d+)')
# Extract first character/letter from Cabin column
df['cabin_cat'] = df['Cabin'].str[0]
# Extract last part of Ticket value
# Usually ticket number
df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
# Convert extracted ticket number into numeric type
# Invalid values become NaN
df['ticket_num'] = pd.to_numeric(
    df['ticket_num'],
    errors='coerce',
    downcast='integer'
)
# Extract first part of Ticket value
# Usually ticket category/prefix
df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])
# If extracted value is only digits, replace with NaN
# Keep only text categories
df['ticket_cat'] = np.where(
    df['ticket_cat'].str.isdigit(),
    np.nan,
    df['ticket_cat']
)