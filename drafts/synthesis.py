import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm

# Set random seed for reproducibility
np.random.seed(42)

# Generate hourly timestamps for 3 months
date_range = pd.date_range(start="2025-01-01", end="2025-03-31", freq="H")

# Define hospital characteristics
hospital_configs = {
    "Hospital_1": {"size": "large", "base_arrivals": 8, "capacity": 100},
    "Hospital_2": {"size": "medium", "base_arrivals": 6, "capacity": 75},
    "Hospital_3": {"size": "large", "base_arrivals": 7, "capacity": 90},
    "Hospital_4": {"size": "small", "base_arrivals": 4, "capacity": 50},
    "Hospital_5": {"size": "medium", "base_arrivals": 5, "capacity": 70}
}

def generate_hourly_pattern():
    """Generate hourly arrival rate multipliers based on typical ED patterns"""
    # Peak hours are typically 10am-2pm and 6pm-10pm
    base_pattern = np.ones(24)
    morning_peak = norm.pdf(np.arange(24), loc=12, scale=2)
    evening_peak = norm.pdf(np.arange(24), loc=20, scale=2)
    
    # Normalize and combine patterns
    hourly_pattern = base_pattern + 5 * (morning_peak + evening_peak)
    return hourly_pattern / np.mean(hourly_pattern)

def generate_weekly_pattern():
    """Generate weekly arrival rate multipliers"""
    # Weekend vs weekday pattern (Mon=0, Sun=6)
    return np.array([1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 1.2])

def generate_hospital_data(hospital_id, config):
    num_records = len(date_range)
    hourly_pattern = generate_hourly_pattern()
    weekly_pattern = generate_weekly_pattern()
    
    # Generate base arrival rates with temporal patterns
    hour_of_day = date_range.hour
    day_of_week = date_range.dayofweek
    
    # Combine temporal patterns
    temporal_multiplier = (hourly_pattern[hour_of_day] * 
                         weekly_pattern[day_of_week])
    
    # Generate correlated random variables
    base_volume = np.random.normal(loc=1, scale=0.1, size=num_records)
    
    # Generate ambulance arrivals with temporal patterns
    mean_arrivals = config["base_arrivals"] * temporal_multiplier * base_volume
    ambulance_arrivals = np.random.poisson(mean_arrivals)
    
    # Generate occupancy that correlates with arrivals
    base_occupancy = 0.6 + (0.2 * base_volume)  # Base occupancy between 60-80%
    occupancy_noise = np.random.normal(0, 0.05, num_records)
    occupancy = np.clip(base_occupancy + occupancy_noise, 0.4, 1.0)
    
    # Generate waiting times that correlate with occupancy
    base_waiting_time = 20 + (100 * occupancy)  # Base waiting time increases with occupancy
    waiting_time_noise = np.random.exponential(10, num_records)
    waiting_times = np.clip(base_waiting_time + waiting_time_noise, 0, 360)  # Cap at 6 hours
    
    # Generate severity scores with time-dependent distribution
    severity_base = 3 + np.random.lognormal(mean=0.5, sigma=0.4, size=num_records)
    # Higher severity during night hours (22:00-06:00)
    night_hours = (hour_of_day >= 22) | (hour_of_day <= 6)
    severity_scores = np.where(night_hours, 
                             severity_base * 1.2,  # Higher severity at night
                             severity_base)
    severity_scores = np.clip(severity_scores, 1, 10).astype(int)
    
    # Calculate derived metrics
    available_beds = config["capacity"]
    current_patients = (occupancy * available_beds).astype(int)
    
    df = pd.DataFrame({
        "Timestamp": date_range,
        "Hospital_ID": hospital_id,
        "Ambulance_Arrivals": ambulance_arrivals,
        "Patient_Waiting_Time_Minutes": waiting_times.astype(int),
        "A&E_Bed_Occupancy": occupancy,
        "Patient_Severity_Score": severity_scores,
        "Current_Patients": current_patients,
        "Available_Beds": available_beds,
        "Hour": hour_of_day,
        "DayOfWeek": day_of_week
    })
    
    return df

# Generate data for all hospitals
synthetic_data = pd.concat([
    generate_hospital_data(hospital_id, config) 
    for hospital_id, config in hospital_configs.items()
])

# Add some seasonal effects
synthetic_data['Month'] = synthetic_data['Timestamp'].dt.month
winter_months = [1, 2, 12]
synthetic_data.loc[synthetic_data['Month'].isin(winter_months), 'Ambulance_Arrivals'] *= 1.2
synthetic_data.loc[synthetic_data['Month'].isin(winter_months), 'Patient_Severity_Score'] += 1

# Sort by timestamp and hospital
synthetic_data = synthetic_data.sort_values(['Timestamp', 'Hospital_ID']).reset_index(drop=True)

# Save the enhanced dataset
synthetic_data.to_csv("synthetic_hospital_data2.csv", index=False)