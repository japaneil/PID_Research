import pandas as pd

chunksize = 1_000_000
i = 0
for chunk in pd.read_csv("pid_parallel_results.csv", chunksize=chunksize):
    chunk.to_csv(f"pid_results_part_{i}.csv", index=False)
    i += 1
