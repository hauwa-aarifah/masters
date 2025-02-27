# NHS A&E Department Simulation System

## 🏥 Overview
This system simulates NHS Accident & Emergency (A&E) department operations, generating synthetic data that models patient flow, resource utilization, and crisis events. The simulation incorporates real-world patterns and NHS operational targets to produce realistic data for analysis and prediction.

## 🌟 Key Features

### Data Generation
- **Patient Arrivals**: Models hourly, daily, and seasonal patterns
- **Severity Scores**: 10-point scale with time-dependent distribution
- **Staff Scheduling**: Realistic shift patterns with handover periods
- **Resource Management**: Bed capacity and staff level modeling
- **Crisis Events**: Simulation of overcrowding and resource constraints

### Analysis Capabilities
- **Performance Metrics**: NHS target compliance tracking
- **Temporal Analysis**: Hourly, daily, and seasonal patterns
- **Hospital Comparisons**: Cross-facility performance analysis
- **Crisis Detection**: Multi-factor overcrowding analysis

## 📊 System Components

### 1. EnhancedHospitalSimulation
Main simulation class for generating synthetic hospital data.

```python
simulator = EnhancedHospitalSimulation(
    start_date="2025-01-01",
    end_date="2025-03-31"
)
data = simulator.generate_full_dataset()
```

Key Methods:
- `generate_handover_delays()`: Models ambulance handover times
- `generate_severity_scores()`: Creates realistic patient severity distribution
- `generate_staff_schedule()`: Simulates staff shift patterns
- `introduce_resource_shocks()`: Models unexpected resource constraints

### 2. HospitalDataAnalyzer
Analysis and validation class for hospital data.

```python
analyzer = HospitalDataAnalyzer(data)
metrics = analyzer.calculate_performance_metrics()
patterns = analyzer.analyze_temporal_patterns()
```

Key Features:
- Performance metrics calculation
- NHS target compliance checking
- Temporal pattern analysis
- Hospital-specific performance analysis

### 3. HospitalDataVisualizer
Visualization suite for analyzing hospital data.

```python
visualizer = HospitalDataVisualizer(data)
visualizer.plot_temporal_patterns()
visualizer.plot_hospital_comparison()
visualizer.plot_crisis_analysis()
```

Generated Visualizations:
- Temporal pattern plots
- Hospital performance comparisons
- Crisis event analysis
- Resource utilization charts

## 💻 Usage Guide

### 1. Basic Usage
```python
from hospital_system import main

# Generate data and create all visualizations
synthetic_data, summary = main()
```

### 2. Custom Data Generation
```python
# Initialize simulator with custom date range
simulator = EnhancedHospitalSimulation(
    start_date="2025-01-01",
    end_date="2025-12-31"
)

# Generate data
data = simulator.generate_full_dataset()

# Save to CSV
data.to_csv('custom_hospital_data.csv', index=False)
```

### 3. Custom Analysis
```python
# Initialize analyzer
analyzer = HospitalDataAnalyzer(data)

# Get performance metrics
metrics = analyzer.calculate_performance_metrics()

# Analyze temporal patterns
patterns = analyzer.analyze_temporal_patterns()

# Analyze hospital performance
hospital_metrics = analyzer.analyze_hospital_performance()
```

### 4. Custom Visualization
```python
# Initialize visualizer
visualizer = HospitalDataVisualizer(data)

# Create specific plots
visualizer.plot_temporal_patterns()
visualizer.plot_hospital_comparison()
visualizer.plot_crisis_analysis()

# Generate summary report
summary = visualizer.generate_summary_report()
```

## 📈 Output Files

### 1. Data Files
- `synthetic_hospital_data.csv`: Raw synthetic data
- `visualizations/summary_report.json`: Comprehensive analysis summary

### 2. Visualization Files
- `visualizations/temporal_patterns.png`: Temporal analysis plots
- `visualizations/hospital_comparison.png`: Hospital comparison plots
- `visualizations/crisis_analysis.png`: Crisis event analysis

## 🎯 NHS Targets Implementation

The system models key NHS operational targets:
- **4-Hour Standard**: 95% of patients should be processed within 4 hours
- **Ambulance Handovers**: 15-minute target for handovers
- **Bed Occupancy**: Target of 85% optimal occupancy

## 🔧 Configuration

### Hospital Configurations
```python
hospital_configs = {
    "Royal London": {
        "size": "large",
        "base_arrivals": 12,
        "bed_capacity": 120,
        "staff_day": 45,
        "staff_night": 30,
        "treatment_rooms": 25,
        "icu_beds": 20
    },
    # Add more hospitals as needed
}
```

### NHS Targets Configuration
```python
nhs_targets = {
    'waiting_time_target': 240,  # 4-hour target in minutes
    'handover_target': 15,       # 15-minute target
    'occupancy_target': 0.85     # 85% occupancy target
}
```

## 📋 Data Dictionary

### Generated Features
- `Timestamp`: Date and time of record
- `Hospital_ID`: Hospital identifier
- `Ambulance_Arrivals`: Number of ambulance arrivals
- `Ambulance_Handover_Delay`: Handover delay in minutes
- `Patient_Waiting_Time_Minutes`: Patient wait time
- `A&E_Bed_Occupancy`: Bed occupancy rate (0-1)
- `Patient_Severity_Score`: Severity score (1-10)
- `Staff_Levels`: Available staff count
- `Overcrowding_Event`: Crisis event flag (True/False)

## 🔍 Validation Methods

The system includes various validation methods:
1. Distribution validation for key metrics
2. Temporal pattern validation
3. NHS target compliance checking
4. Crisis event pattern validation

## 🚀 Future Improvements

Planned enhancements:
1. Machine learning integration for predictive analytics
2. Real-time data streaming capabilities
3. Advanced resource optimization algorithms
4. Integration with actual NHS data feeds

## ⚠️ Error Handling

The system includes robust error handling for:
- Data type conversions
- JSON serialization
- File I/O operations
- Configuration validation

## 📚 Dependencies

Required Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
