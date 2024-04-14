import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = "/Users/gizemou/Desktop/datathon/Datathon Data/Nutrition_Physical_Activity_and_Obesity_Data.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Print out the column names to verify them
print(df.columns.tolist())

df = df[df['Stratification1'] == 'Non-Hispanic Black']
# Use this line to select the desired feature

question_ids = ['Q058']
df_filtered = df[df['QuestionID'].isin(question_ids) & df['Data_Value'].notna() & df['Sample_Size'].notna()]

# Calculate the weighted mean for each 'QuestionID'
weighted_means = df_filtered.groupby('QuestionID').apply(
    lambda x: (x['Data_Value'] * x['Sample_Size']).sum() / x['Sample_Size'].sum()
)

# Reset index to use in the bar plot
weighted_means = weighted_means.reset_index(name='Weighted Mean')
print(weighted_means)