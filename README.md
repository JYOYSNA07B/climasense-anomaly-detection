# ClimaSense: Automated Climate Anomaly Detection Using Random Forest & Bidirectional LSTM

> ML pipeline for detecting climate anomalies using Random Forest Classifier with threshold-based labeling and a Bidirectional LSTM hybrid for temperature forecasting. Built on ERA5-style features including t2m, wind vectors, sea level pressure, and geospatial coordinates.

---

## 📌 Project Overview

ClimaSense is a dual-pipeline machine learning system designed for climate intelligence:

- **Pipeline 1 – Anomaly Detection:** Uses a Random Forest Classifier to detect statistically extreme temperature events based on quantile threshold-based labeling (top/bottom 3.2%).
- **Pipeline 2 – Temperature Forecasting:** Uses a Bidirectional LSTM + Random Forest Regressor hybrid to forecast 2-metre air temperature (t2m) using historical climate sequences.

The combined approach delivers reliable climate anomaly identification and strengthens future climate prediction capabilities — supporting environmental monitoring, climate risk evaluation, and climate change response strategies.

---

## 📂 Repository Structure

```
climasense-anomaly-detection/
│
├── anomaly_detection.ipynb        # Random Forest anomaly detection pipeline
├── temperature_forecast.ipynb     # Bidirectional LSTM forecasting pipeline
└── README.md                      # Project documentation
```

---

## 📊 Dataset

- **Source:** [2024 Climate Data India (Kaggle)](https://www.kaggle.com/datasets/shreelearn/2024-climate-data-india)
- **Platform:** Kaggle
- **Type:** Preprocessed & Reduced Dataset (50,000 records)

> ⚠️ The dataset is originally sourced from Kaggle and has been preprocessed, cleaned, and reduced for this project.
> Due to Kaggle's usage policies and file size limits, the dataset is not included in this repository.

---

### 📥 How to Use Dataset

1. Go to the dataset link above  
2. Download the dataset from Kaggle  
3. Rename the file to `50k.csv` (if required)  
4. Upload it into your working environment (Google Colab / Local system)

### 📌 Data Preprocessing

- Reduced dataset from ~200,000 to 50,000 records  
- Removed missing values and duplicates  
- Applied feature selection and correlation filtering  
- Normalized numerical features  
- Generated anomaly labels using quantile thresholds  
---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Environment:** Google Colab
- **Libraries:**

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data processing & manipulation |
| `scikit-learn` | Random Forest, preprocessing, metrics |
| `tensorflow` / `keras` | Bidirectional LSTM model |
| `matplotlib`, `seaborn` | Visualizations & plots |

---

## ⚙️ How to Run

### 1. Open in Google Colab
Upload `anomaly_detection.ipynb` and `temperature_forecast.ipynb` to Google Colab.

### 2. Upload the Dataset
Upload your `50k.csv` file to the Colab session:
```python
from google.colab import files
files.upload()  # Upload 50k.csv
```

### 3. Install Dependencies
```python
!pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### 4. Run Anomaly Detection
Open and run all cells in `anomaly_detection.ipynb`

### 5. Run Temperature Forecasting
Open and run all cells in `temperature_forecast.ipynb`

---

## 📈 Model Performance

### Pipeline 1 — Temperature Anomaly Detection (Random Forest Classifier)

| Metric | Score |
|---|---|
| Training Accuracy | 0.9350 (93.50%) |
| Testing Accuracy | 0.9422 (94.22%) |
| Total Model Accuracy | 0.9386 (93.86%) |
| Precision | 0.9569 (95.69%) |
| Recall | 0.9260 (92.60%) |
| F1 Score | 0.9412 (94.12%) |

> From the dataset, **3,200 anomalies** were identified out of 50,000 records. Anomalies mainly appeared on `2024-02-01`, showing extremely high t2m values in certain latitudes.

### Pipeline 2 — Climate Change Prediction (Bi-LSTM + Random Forest Regressor)

| Metric | Value |
|---|---|
| LSTM Training Loss (MSE) | 0.0047 |
| LSTM Testing Loss (MSE) | 0.0042 |
| RF Training Loss (MSE) | 0.0084 |
| RF Testing Loss (MSE) | 0.0016 |
| RMSE | 0.04039 |
| MAE | 0.02381 |

> Predicted temperatures for January 2025 ranged around **300 K (~27°C)**, with latitude fixed at 13.49° across varying longitudes.

---

## 📉 Visualizations

| Figure | Description |
|---|---|
| Fig 1 | **Scatter Plot** — Latitude vs Temperature with anomalies (red = anomaly, blue = normal) |
| Fig 2 | **Feature Correlation Heatmap** — t2m shows strong negative correlation with latitude (–0.93) |
| Fig 3 | **Time-Series Plot** — Predicted (red) vs Actual (blue) temperature over time |
| Fig 4 | **Bar Chart** — Model performance comparison (LSTM vs Random Forest metrics) |

---

## 🔍 Methodology

### Pipeline 1 — Anomaly Detection
1. Load and preprocess dataset (handle missing values, remove duplicates)
2. Convert `time` column to datetime format
3. Drop highly correlated features (threshold > 0.85)
4. Apply Gaussian noise to `t2m` to reduce overfitting
5. Label anomalies using top/bottom 3.2% quantile thresholds
6. Balance dataset (3,200 anomalies : 3,200 normal samples)
7. Train Random Forest Classifier with optimized hyperparameters
8. Apply optimal classification threshold using ROC curve analysis

### Pipeline 2 — Temperature Forecasting
1. Normalize features using MinMaxScaler
2. Create time-step sequences (48 time steps)
3. Train Bidirectional LSTM with LayerNormalization and Dropout layers
4. Apply EarlyStopping and ReduceLROnPlateau callbacks
5. Use LSTM predictions as input to a Random Forest Regressor
6. Inverse transform predictions back to Kelvin scale

---

## 🏗️ Model Architecture

### Bidirectional LSTM
```
Input Layer       → (timesteps=48, features=5)
Bi-LSTM Layer 1   → 128 units, return_sequences=True
LayerNormalization + Dropout (0.25)
Bi-LSTM Layer 2   → 64 units, return_sequences=False
LayerNormalization + Dropout (0.20)
Dense Layer       → 16 units, ReLU
Dense Layer       → 8 units, ReLU
Output Layer      → 1 unit, Linear
```

---

## 🔑 Key Findings

- Temperature (t2m) shows a **strong negative correlation with latitude (–0.93)**, confirming geographical influence on climate behaviour.
- The RFC model achieved **94.22% testing accuracy** with **95.69% precision**, making it suitable for real-time anomaly monitoring.
- The Bi-LSTM + RF hybrid achieved an **RMSE of 0.04039** and **MAE of 0.02381**, demonstrating reliable climate forecasting.
- Combining deep learning (LSTM) with ensemble methods (Random Forest) produced better results than either model alone.

---

## 🚀 Future Scope

- Integrate additional climate variables such as precipitation, humidity, and wind patterns
- Implement real-time streaming data for live anomaly detection and early warning systems
- Apply spatial analysis and geostatistical models for regional climate hotspot detection
- Explore Transformer-based architectures and GANs for improved spatiotemporal predictions
- Deploy on scalable cloud platforms for global climate research

---

## 📚 References

1. Chin, S., & Lloyd, V. (2024). Predicting climate change using an autoregressive LSTM model. *Frontiers in Environmental Science*, 12, 1301343.
2. Chen, L., et al. (2023). Machine learning methods in weather and climate applications: A survey. *Applied Sciences*, 13(21), 12019.
3. Materia, S., et al. (2024). Artificial intelligence for climate prediction of extremes. *Wiley Interdisciplinary Reviews: Climate Change*, 15(6), e914.
4. Vázquez-Ramírez, S., et al. (2023). An analysis of climate change based on machine learning and an endoreversible model. *Mathematics*, 11(14), 3060.

---

## 📃 License

This project is licensed under the MIT License.

---

## 🙋 Author

**Your Name**  
📧 jyojyotsna72@gmail.com  
🔗 www.linkedin.com/in/jyotsna-b-586498373| https://github.com/JYOYSNA07B
