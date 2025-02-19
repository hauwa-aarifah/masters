import numpy as np
import pandas as pd

# Set random seed
np.random.seed(42)
print("hello")

# Generate hourly timestamps
date_range = pd.date_range(start="2025-01-01", end="2025-03-31", freq="H")

# Define hospital IDs
hospitalids = [f"Hospital{i+1}" for i in range(5)]

# Function to generate synthetic hospital data
def generate_hospital_data(hospital_id):
    num_records = len(date_range)

    ambulance_arrivals = np.random.poisson(lam=5, size=num_records)
    waiting_times = np.random.exponential(scale=30, size=num_records).astype(int)
    occupancy = np.clip(np.random.normal(loc=0.75, scale=0.1, size=num_records), 0.5, 1.0)
    severity_scores = np.clip(np.random.lognormal(mean=1.5, sigma=0.5, size=num_records), 1, 10).astype(int)

    df = pd.DataFrame({
        "Timestamp": date_range,
        "Hospital_ID": hospital_id,
        "Ambulance_Arrivals": ambulance_arrivals,
        "Patient_Waiting_Time_Minutes": waiting_times,
        "A&E_Bed_Occupancy": occupancy,
        "Patient_Severity_Score": severity_scores
    })

    return df

# Generate data for all hospitals
synthetic_hospital_data = pd.concat([generate_hospital_data(hospital) for hospital in hospital_ids])

# Save for further validation
synthetic_hospital_data.to_csv("synthetic_hospital_data.csv", index=False)