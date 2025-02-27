import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class EnhancedHospitalSimulation:
    """Main simulation class for generating synthetic hospital data"""
    
    def __init__(self, start_date="2025-01-01", end_date="2025-03-31"):
        self.date_range = pd.date_range(start=start_date, end=end_date, freq="h")  # Changed from "H" to "h"
        self.hospital_configs = {
            "Royal London": {
                "size": "large",
                "region": "London",
                "base_arrivals": 12,
                "bed_capacity": 120,
                "staff_day": 45,
                "staff_night": 30,
                "treatment_rooms": 25,
                "icu_beds": 20,
                "base_performance": 0.713  # London region performance
            },
            "St Thomas": {
                "size": "large",
                "region": "London", 
                "base_arrivals": 10,
                "bed_capacity": 100,
                "staff_day": 40,
                "staff_night": 25,
                "treatment_rooms": 20,
                "icu_beds": 18,
                "base_performance": 0.713  # London region performance
            },
            "Manchester Royal": {
                "size": "large",
                "region": "North West",
                "base_arrivals": 11,
                "bed_capacity": 110,
                "staff_day": 42,
                "staff_night": 28,
                "treatment_rooms": 22,
                "icu_beds": 19,
                "base_performance": 0.678 # North West region performance
            }
        }

    def generate_handover_delays(self, occupancy, hour_of_day, month):
        """Generate ambulance handover delays using bimodal distribution"""
        # Base delays based on NHS data
        base_delays = np.zeros(len(occupancy))
        
        # Normal hours (default scenario)
        base_delays = np.where(
            (hour_of_day >= 9) & (hour_of_day <= 17),
            np.random.normal(25, 10),  # 20-35 minutes
            base_delays
        )
        
        # Peak hour adjustments
        peak_hours = ((hour_of_day >= 5) & (hour_of_day <= 8)) | \
                     ((hour_of_day >= 17) & (hour_of_day <= 20))
        base_delays[peak_hours] = np.random.normal(40, 15)  # 35-50 minutes
        
        # Winter months and crisis periods
        winter_months = [12, 1, 2]
        is_winter = np.isin(month, winter_months)
        base_delays[is_winter] *= 1.3  # 30% increase in delays
        
        # Occupancy impact
        high_occupancy = occupancy > 0.95
        base_delays[high_occupancy] = np.random.normal(60, 30)  # Crisis periods
        
        return np.clip(base_delays, 5, 180)

    def generate_severity_scores(self, hour_of_day, month, arrivals):
        """Generate severity scores with time-dependent distribution"""
        base_probs = [0.05, 0.10, 0.25, 0.30, 0.15, 0.07, 0.05, 0.02, 0.01]
        
        severity_scores = np.random.choice(
            range(1, 10), 
            size=len(arrivals), 
            p=base_probs
        )
        
        # Increase severity during night hours
        night_hours = (hour_of_day >= 22) | (hour_of_day <= 6)
        night_boost = np.random.choice(
            [0, 1, 2], 
            size=sum(night_hours), 
            p=[0.6, 0.3, 0.1]
        )
        severity_scores[night_hours] += night_boost

        # Winter months additional severity
        winter_months = [12, 1, 2]
        is_winter = np.isin(month, winter_months)
        severity_scores[is_winter] += np.random.choice(
            [0, 1], 
            size=sum(is_winter), 
            p=[0.7, 0.3]
        )
        
        return np.clip(severity_scores, 1, 10)

    def generate_staff_schedule(self, num_hours, day_staff, night_staff, month):
        """Generate staff levels with winter and seasonal variations"""
        schedule = np.zeros(num_hours)
        
        for i in range(0, num_hours, 24):
            # Day shift (8am-8pm)
            schedule[i+8:i+20] = day_staff
            # Night shift (8pm-8am)
            schedule[i+20:i+24] = night_staff
            if i+24 < num_hours:
                schedule[i+0:i+8] = night_staff
            
            # Handover periods
            schedule[i+7:i+9] *= 0.8  # Morning handover
            schedule[i+19:i+21] *= 0.8  # Evening handover
        
        # Winter months staff reduction
        winter_months = [12, 1, 2]
        if month[0] in winter_months:
            schedule *= 0.9  # 10% staff reduction during winter
        
        # Add random variation
        variation = np.random.normal(1, 0.05, num_hours)
        return np.round(schedule * variation)

    def introduce_resource_shocks(self, staff_levels, dates):
        """Introduce random resource shock events"""
        daily_dates = pd.date_range(dates[0], dates[-1], freq='D')
        num_days = len(daily_dates)
        
        shock_days = np.random.choice(
            range(num_days), 
            size=int(0.05 * num_days),
            replace=False
        )
        
        shock_calendar = pd.Series(index=daily_dates, data=1.0)
        shock_calendar.iloc[shock_days] = 0.7
        
        hourly_shocks = shock_calendar.reindex(dates, method='ffill')
        shocked_staff = staff_levels * hourly_shocks
        
        return np.clip(shocked_staff, 5, None)

    def generate_arrival_pattern(self, base_rate, hour, day, month, region=None):
        """Generate arrival patterns with regional and temporal nuances"""
        # Hourly multiplier with enhanced peaks
        hourly_mult = 1 + 0.4 * np.sin(np.pi * (hour - 10) / 12)
        
        # Weekend effect
        weekend_mult = 1 + 0.3 * (day >= 5)
        
        # Winter months surge
        winter_months = [12, 1, 2]
        winter_mult = 1 + 0.3 * np.isin(month, winter_months)
        
        # Regional variations
        regional_multipliers = {
            "London": 1.1,
            "North West": 0.95
        }
        region_mult = regional_multipliers.get(region, 1.0)
        
        return base_rate * hourly_mult * weekend_mult * winter_mult * region_mult

    def calculate_occupancy(self, arrivals, severity, capacity, month):
        """Calculate bed occupancy considering patient severity"""
        # Length of stay multiplier based on severity
        los_multiplier = np.clip(severity / 5, 0.5, 2.0)

        # Current patients tracking
        current_patients = np.zeros(len(arrivals))
        
        # Winter months increase occupancy
        winter_months = [12, 1, 2]
        winter_boost = 1.05 if np.isin(month[0], winter_months) else 1.0
        
        for t in range(len(arrivals)):
            stay_duration = int(4 * los_multiplier[t])
            end_idx = min(t + stay_duration, len(arrivals))
            current_patients[t:end_idx] += arrivals[t]
        
        # Clip occupancy with winter boost and realistic ceiling
        return np.clip(current_patients / (capacity * winter_boost), 0, 1.1)

    def calculate_waiting_times(self, occupancy, staff_levels, severity, hour_of_day, month):
        """Calculate waiting times with NHS-realistic performance"""
        # Base waiting time calculations
        base_waiting_time = 60 + (120 * occupancy)
        staff_factor = np.clip(1 + (1 - staff_levels/np.mean(staff_levels)), 0.8, 2.0)
        severity_factor = 1 / (severity / 3)
        
        # Time of day adjustments
        night_hours = (hour_of_day >= 22) | (hour_of_day <= 6)
        
        # Detect weekend 
        # Use the day of week index from the date range
        weekend_hours = [d % 7 >= 5 for d in range(len(hour_of_day))]
        
        # Winter months additional delay
        winter_months = [12, 1, 2]
        winter_months_mask = np.isin(month, winter_months)
        
        # Waiting time calculations
        waiting_times = base_waiting_time * staff_factor * severity_factor
        
        # Night hour penalty
        waiting_times[night_hours] *= 1.5
        
        # Weekend penalty
        waiting_times[weekend_hours] *= 1.3
        
        # Winter months penalty
        waiting_times[winter_months_mask] *= 1.4
        
        # Add noise and clip
        waiting_time_noise = np.random.exponential(15, len(occupancy))
        
        return np.clip(waiting_times + waiting_time_noise, 0, 480)  # Max 8 hours

    def generate_hospital_data(self, hospital_id, config):
        """Generate synthetic data for a single hospital"""
        num_records = len(self.date_range)
        
        # Generate staff levels
        base_staff = self.generate_staff_schedule(
            num_records, 
            config['staff_day'],
            config['staff_night'],
            self.date_range.month
        )
        staff_levels = base_staff
        
        # Generate temporal patterns
        hour_of_day = self.date_range.hour
        day_of_week = self.date_range.dayofweek
        month = self.date_range.month
        
        # Generate arrivals
        mean_arrivals = self.generate_arrival_pattern(
            config["base_arrivals"],
            hour_of_day,
            day_of_week,
            month,
            config.get('region', 'default')
        )
        ambulance_arrivals = np.random.poisson(mean_arrivals)
        
        # Generate other metrics
        severity_scores = self.generate_severity_scores(hour_of_day, month, ambulance_arrivals)
        occupancy = self.calculate_occupancy(ambulance_arrivals, severity_scores, config["bed_capacity"], month)
        handover_delays = self.generate_handover_delays(occupancy, hour_of_day, month)
        waiting_times = self.calculate_waiting_times(
            occupancy, 
            staff_levels, 
            severity_scores, 
            hour_of_day,  # Pass hour_of_day directly 
            month
        )
        
        # Detect overcrowding with NHS-realistic thresholds
        overcrowding = (occupancy > 0.95) & (waiting_times > 240)
        
        return pd.DataFrame({
            "Timestamp": self.date_range,
            "Hospital_ID": hospital_id,
            "Ambulance_Arrivals": ambulance_arrivals,
            "Ambulance_Handover_Delay": handover_delays,
            "Patient_Waiting_Time_Minutes": waiting_times.astype(int),
            "A&E_Bed_Occupancy": occupancy,
            "Patient_Severity_Score": severity_scores,
            "Staff_Levels": staff_levels,
            "Overcrowding_Event": overcrowding,
            "Hour": hour_of_day,
            "DayOfWeek": day_of_week,
            "Month": month
        })

    def generate_full_dataset(self):
        """Generate complete dataset for all hospitals"""
        all_data = []
        for hospital_id, config in self.hospital_configs.items():
            print(f"Generating data for {hospital_id}...")
            hospital_data = self.generate_hospital_data(hospital_id, config)
            all_data.append(hospital_data)
        
        synthetic_data = pd.concat(all_data)
        return synthetic_data.sort_values(['Timestamp', 'Hospital_ID']).reset_index(drop=True)

class HospitalDataAnalyzer:
    """Analysis and validation class for hospital data"""
    
    def __init__(self, data):
        self.data = data
        self.nhs_targets = {
            'waiting_time_target': 240,  # 4-hour target
            'handover_target': 15,  # 15-minute target
            'occupancy_target': 0.85  # 85% occupancy target
        }

    def calculate_performance_metrics(self):
        """Calculate key performance metrics"""
        metrics = {
            'waiting_time_compliance': (
                self.data['Patient_Waiting_Time_Minutes'] <= 
                self.nhs_targets['waiting_time_target']
            ).mean() * 100,
            
            'handover_compliance': (
                self.data['Ambulance_Handover_Delay'] <= 
                self.nhs_targets['handover_target']
            ).mean() * 100,
            
            'occupancy_compliance': (
                self.data['A&E_Bed_Occupancy'] <= 
                self.nhs_targets['occupancy_target']
            ).mean() * 100,
            
            'average_metrics': {
                'waiting_time': self.data['Patient_Waiting_Time_Minutes'].mean(),
                'occupancy': self.data['A&E_Bed_Occupancy'].mean(),
                'handover_delay': self.data['Ambulance_Handover_Delay'].mean(),
                'severity': self.data['Patient_Severity_Score'].mean()
            },
            
            'crisis_events': {
                'total': self.data['Overcrowding_Event'].sum(),
                'percentage': self.data['Overcrowding_Event'].mean() * 100
            }
        }
        
        return metrics

    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in the data"""
        patterns = {
            'hourly': self.data.groupby('Hour')['Ambulance_Arrivals'].mean(),
            'daily': self.data.groupby('DayOfWeek')['Ambulance_Arrivals'].mean(),
            'monthly': self.data.groupby('Month')['Ambulance_Arrivals'].mean()
        }
        return patterns

    def analyze_hospital_performance(self):
        """Analyze performance by hospital"""
        hospital_metrics = {}
        
        for hospital in self.data['Hospital_ID'].unique():
            hospital_data = self.data[self.data['Hospital_ID'] == hospital]
            hospital_metrics[hospital] = {
                'waiting_time': hospital_data['Patient_Waiting_Time_Minutes'].mean(),
                'occupancy': hospital_data['A&E_Bed_Occupancy'].mean(),
                'crisis_rate': hospital_data['Overcrowding_Event'].mean() * 100
            }
        
        return hospital_metrics

class HospitalDataVisualizer:
    """Visualization class for hospital data"""
    
    def __init__(self, data):
        self.data = data
        self.output_dir = 'final_visualizations'
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_temporal_patterns(self):
        """Plot temporal patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Hourly patterns
        sns.lineplot(
            data=self.data, 
            x='Hour', 
            y='Ambulance_Arrivals',
            ax=axes[0,0]
        )
        axes[0,0].set_title('Hourly Arrival Pattern')
        
        # Daily patterns
        sns.boxplot(
            data=self.data,
            x='DayOfWeek',
            y='Ambulance_Arrivals',
            ax=axes[0,1]
        )
        axes[0,1].set_title('Daily Arrival Pattern')
        
        # Occupancy patterns
        sns.lineplot(
            data=self.data,
            x='Hour',
            y='A&E_Bed_Occupancy',
            ax=axes[1,0]
        )
        axes[1,0].set_title('Hourly Occupancy Pattern')
        
        # Waiting time patterns
        sns.boxplot(
            data=self.data,
            x='Hour',
            y='Patient_Waiting_Time_Minutes',
            ax=axes[1,1]
        )
        axes[1,1].set_title('Waiting Times by Hour')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/temporal_patterns.png')
        plt.close()

    def plot_hospital_comparison(self):
        """Plot hospital comparisons"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Waiting times by hospital
        sns.boxplot(
            data=self.data,
            x='Hospital_ID',
            y='Patient_Waiting_Time_Minutes',
            ax=axes[0,0]
        )
        axes[0,0].set_title('Waiting Times by Hospital')
        plt.xticks(rotation=45)
        
        # Occupancy by hospital
        sns.boxplot(
            data=self.data,
            x='Hospital_ID',
            y='A&E_Bed_Occupancy',
            ax=axes[0,1]
        )
        axes[0,1].set_title('Occupancy by Hospital')
        plt.xticks(rotation=45)
        
        # Crisis events by hospital
        crisis_by_hospital = self.data.groupby('Hospital_ID')['Overcrowding_Event'].mean()
        sns.barplot(
            x=crisis_by_hospital.index,
            y=crisis_by_hospital.values,
            ax=axes[1,0]
        )
        axes[1,0].set_title('Crisis Event Rate by Hospital')
        plt.xticks(rotation=45)
        
        # Severity by hospital
        sns.boxplot(
            data=self.data,
            x='Hospital_ID',
            y='Patient_Severity_Score',
            ax=axes[1,1]
        )
        axes[1,1].set_title('Patient Severity by Hospital')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/hospital_comparison.png')
        plt.close()

    def plot_crisis_analysis(self):
        """Plot crisis event analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Crisis probability by hour
        crisis_by_hour = self.data.groupby('Hour')['Overcrowding_Event'].mean()
        sns.barplot(
            x=crisis_by_hour.index,
            y=crisis_by_hour.values,
            ax=axes[0,0]
        )
        axes[0,0].set_title('Crisis Probability by Hour')
        
        # Waiting times during crisis
        sns.boxplot(
            data=self.data,
            x='Overcrowding_Event',
            y='Patient_Waiting_Time_Minutes',
            ax=axes[0,1]
        )
        axes[0,1].set_title('Waiting Times: Crisis vs Normal')
        
        # Severity during crisis
        sns.boxplot(
            data=self.data,
            x='Overcrowding_Event',
            y='Patient_Severity_Score',
            ax=axes[1,0]
        )
        axes[1,0].set_title('Patient Severity: Crisis vs Normal')
        
        # Staff levels during crisis
        sns.boxplot(
            data=self.data,
            x='Overcrowding_Event',
            y='Staff_Levels',
            ax=axes[1,1]
        )
        axes[1,1].set_title('Staff Levels: Crisis vs Normal')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/crisis_analysis.png')
        plt.close()

    def generate_summary_report(self):
        """Generate a comprehensive summary report with proper JSON serialization"""
        analyzer = HospitalDataAnalyzer(self.data)
        
        # Convert temporal patterns to standard Python types
        temporal_patterns = analyzer.analyze_temporal_patterns()
        serializable_patterns = {
            key: {str(k): float(v) for k, v in values.items()}
            for key, values in temporal_patterns.items()
        }
        
        # Get performance metrics and convert to standard Python types
        performance_metrics = analyzer.calculate_performance_metrics()
        
        # Convert hospital performance metrics
        hospital_performance = analyzer.analyze_hospital_performance()
        serializable_hospital_performance = {
            hospital: {
                metric: float(value)
                for metric, value in metrics.items()
            }
            for hospital, metrics in hospital_performance.items()
        }
        
        summary = {
            'data_overview': {
                'total_records': int(len(self.data)),
                'date_range': {
                    'start': self.data['Timestamp'].min().strftime('%Y-%m-%d'),
                    'end': self.data['Timestamp'].max().strftime('%Y-%m-%d')
                },
                'hospitals': self.data['Hospital_ID'].unique().tolist()
            },
            'performance_metrics': {
                'waiting_time_compliance': float(performance_metrics['waiting_time_compliance']),
                'handover_compliance': float(performance_metrics['handover_compliance']),
                'occupancy_compliance': float(performance_metrics['occupancy_compliance']),
                'average_metrics': {
                    key: float(value)
                    for key, value in performance_metrics['average_metrics'].items()
                },
                'crisis_events': {
                    'total': int(performance_metrics['crisis_events']['total']),
                    'percentage': float(performance_metrics['crisis_events']['percentage'])
                }
            },
            'hospital_performance': serializable_hospital_performance,
            'temporal_patterns': serializable_patterns
        }
        
        # Save summary report
        with open(f'{self.output_dir}/summary_report.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        return summary

def main():
    """Main function to run the complete hospital system"""
    print("Starting hospital system simulation...")
    
    # Generate synthetic data
    simulator = EnhancedHospitalSimulation()
    synthetic_data = simulator.generate_full_dataset()
    
    # Save raw data
    synthetic_data.to_csv('final_synthetic_hospital_data.csv', index=False)
    print("Synthetic data generated and saved.")
    
    # Create visualizations
    print("Generating visualizations...")
    visualizer = HospitalDataVisualizer(synthetic_data)
    visualizer.plot_temporal_patterns()
    visualizer.plot_hospital_comparison()
    visualizer.plot_crisis_analysis()
    
    # Generate summary report
    print("Generating summary report...")
    summary = visualizer.generate_summary_report()
    
    print("\nProcess complete!")
    print("- Raw data saved as 'final_synthetic_hospital_data.csv'")
    print("- Visualizations saved in 'final_visualizations' directory")
    print("- Summary report saved as 'final_visualizations/summary_report.json'")
    
    return synthetic_data, summary

if __name__ == "__main__":
    synthetic_data, summary = main()