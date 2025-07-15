import pandas as pd

# Load part 2
df = pd.read_csv("pid_results_part_2.csv")

# Remove rows where TSE is inf (as float or string)
df = df[~df['TSE'].isin([float('inf'), 'inf', 'Infinity'])]

# Save cleaned version
df.to_csv("clean_pid_results_part_2.csv", index=False)

print("✅ Cleaned file saved as 'clean_pid_results_part_2.csv'")
