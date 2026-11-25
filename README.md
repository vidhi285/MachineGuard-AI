# MachineGuard AI
### Industrial Machine Failure Prediction and Predictive Maintenance Using Machine Learning

MachineGuard AI is a machine learning-based application designed to analyse industrial machine operating data and identify possible failure conditions. It aims to support predictive maintenance by helping users inspect machine readings, review prediction results and organise maintenance activities.

## Project Overview

Unexpected machine failures can lead to production downtime, increased maintenance costs and operational disruptions. MachineGuard AI uses trained machine learning classification models to analyse sensor readings and predict whether a record resembles a failure or non-failure condition.

The project currently supports two machine datasets:

### 1. MetroPT-3 – Air Compressor
- 1,516,948 sensor records.
- Based on real metro-train air compressor operating data.
- Uses pressure readings, oil temperature, motor current and digital signals.
- Feature engineering includes pressure differences, motor current change and oil temperature change.
- **Selected model: Decision Tree**

### 2. AI4I 2020 – General Industrial Machine
- 10,000 records.
- Synthetic predictive maintenance dataset.
- Uses air temperature, process temperature, rotational speed, torque and tool wear.
- Feature engineering includes temperature difference, power proxy, tool wear category and torque per RPM.
- **Selected model: Random Forest**

## Model Performance

| Evaluation Metric | MetroPT-3 (Decision Tree) | AI4I 2020 (Random Forest) |
|---|---:|---:|
| Accuracy | 95.72% | 98.70% |
| Precision | 82.72% | 86.21% |
| Recall | 71.70% | 73.53% |
| F1-score | 76.82% | 79.37% |

These results are based on the respective held-out test datasets and should not be interpreted as guaranteed performance on every real industrial machine.

## Key Features

- **Machine Monitoring:** Dedicated workspaces for both machine categories.
- **Manual Sensor Input:** Enter machine operating parameters to generate predictions.
- **File Upload:** Upload CSV and Excel files for batch prediction.
- **Data Validation:** Identify missing or invalid sensor values before prediction.
- **Failure Prediction:** Classify records into possible failure and non-failure conditions.
- **Machine Registry:** Register and manage machine information.
- **Prediction History:** Review previously generated prediction results.
- **Risk Analytics:** Explore prediction trends and machine-related insights.
- **Maintenance Planner:** Create, update and manage maintenance tasks.

## System Workflow

```text
Machine Sensor Data / File Upload
              ↓
       Data Preprocessing
              ↓
        Data Validation
              ↓
       Feature Engineering
              ↓
     Trained ML Classification Models
              ↓
       Failure Prediction
              ↓
       Streamlit Application
              ↓
       MongoDB Database
              ↓
 Prediction History & Maintenance Planning
```

## Technology Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Frontend:** Streamlit
- **Database:** MongoDB
- **Data Visualization:** Plotly, Matplotlib
- **Model Storage:** Joblib

## Project Structure

```text
MachineGuard-AI/
│
├── app.py
├── config.py
├── prediction.py
├── database.py
├── requirements.txt
├── env.example
│
├── models/
│   ├── compressor_best_model.joblib
│   └── ai4i_best_model.joblib
│
├── ai4i_model_plots/
├── ai4i_plots/
├── compressor_feature_plots/
├── compressor_model_plots/
├── compressor_plots/
│
├── ai4i_feature_engineering.py
├── ai4i_model_training.py
├── compressor_feature_engineering.py
└── compressor_model_training.py
```

## Installation and Execution

1. Clone the repository:

```bash
git clone https://github.com/vidhi285/MachineGuard-AI.git
```

2. Navigate to the project directory:

```bash
cd MachineGuard-AI
```

3. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Configure the environment variables using `env.example` and ensure MongoDB is available.

6. Run the application:

```bash
streamlit run app.py
```

## Project Objective

The objective of MachineGuard AI is to demonstrate how machine learning can be applied to industrial sensor data to identify possible failure patterns and support data-driven maintenance planning.

## Important Note

The models are trained on example datasets: MetroPT-3 metro-train air compressor data and the synthetic AI4I 2020 dataset. Predictions indicate possible failure patterns; they do not confirm physical machine failures or estimate remaining useful life.

---

**MachineGuard AI – Turning Machine Data into Predictive Maintenance Insights.**
