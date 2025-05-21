import pandas as pd
import numpy as np
import ruptures as rpt
import os
import time

# Function to get user input for parameters
def get_user_input():
    Temp = input("Enter the temperature (e.g., 2100): ")
    tme = input("Enter the time (e.g., 100ns): ")
    cut = int(input("Enter the cut value (e.g., 10): "))
    chunk_size = int(input("Enter the chunk size (e.g., 500): "))
    model = input("Enter the model (e.g., rbf): ")
    cs = input("Enter the cs value (e.g., cs5h): ")
    return Temp, tme, cut, chunk_size, model, cs

# Get user input for parameters
Temp, tme, cut, chunk_size, model, cs = get_user_input()

# Start time
s_time = time.time()

# Load your center of mass data
try:
    data = pd.read_csv(f'cent_mass_{Temp}_c.dat.{tme}', skiprows=30, skipfooter=0000, names=['time', 'x', 'y', 'z'], delim_whitespace=True, engine='python')
    print(data)
except FileNotFoundError:
    print(f"File 'cent_mass_{Temp}_c.dat.{tme}' not found. Please check the file path and name.")
    exit()

# Perform change point detection using Pelt algorithm
all_results = np.empty((0, 3))  # Create an empty array to append the calculated results
com = data[['x', 'y', 'z']].to_numpy()

# Specify the file name
file_path = f'result.rpt.{Temp}.{tme}.chunks.cut{cut}.{cs}'
print("File Path:", file_path)

# Check if the file exists before removing it
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Old file '{file_path}' removed.")

for start in range(0, len(com), chunk_size):
    end = min(start + chunk_size, len(com))
    chunk_data = com[start:end]  # Extracting the current chunk of data
    algo = rpt.Pelt(model=model).fit(chunk_data)  # Fit the model on the current chunk
    result = algo.predict(pen=cut)
    if len(chunk_data) in result:  # To miss the start and end point noise
        result[-1] = np.nan
    else:
        continue
    result = [value for value in result if not np.isnan(value)]
    # Apply chunk_index * chunk_size to each element in result
    chunk_index = start // chunk_size
    result = [value + chunk_index * chunk_size for value in result]
    df = pd.DataFrame(result)
    try:
        df.to_csv(file_path, mode='a', header=False, index=True)  # Write to the file
    except Exception as e:
        print("Error occurred while writing to file:", e)

# End time
e_time = time.time()
t_time = (e_time - s_time) / 60
print(f'Total computation time is {t_time} minutes')