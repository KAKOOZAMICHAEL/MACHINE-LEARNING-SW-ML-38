# Uganda TB Risk Prediction API 🏥

REST API for predicting district-level tuberculosis (TB) risk across Uganda's 146 districts using machine learning.

## 🎯 Overview

This API uses **Isolation Forest** (AUC-ROC: 0.701) to identify high-risk TB districts based on demographic, geographic, and health facility data. It helps health officials allocate resources effectively and plan targeted interventions.

## 🏆 Model Performance

- **Algorithm:** Isolation Forest (500 trees)
- **Performance:** AUC-ROC 0.701 (best of 7 models evaluated)
- **Features:** 13 district-level indicators
- **Model Size:** 4.2 MB
- **Prediction Speed:** 47 ms for 135 districts

## 🚀 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/KAKOOZAMICHAEL/MACHINE-LEARNING-SW-ML-38.git
cd MACHINE-LEARNING-SW-ML-38

# Install dependencies
pip install -r requirements_render.txt

# Start API
uvicorn deployment.app:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: http://localhost:8000

API documentation: http://localhost:8000/docs

### Start Dashboard (Optional)

```bash
streamlit run deployment/dashboard.py
```

Dashboard will be available at: http://localhost:8501

## 📡 API Endpoints

### Health Check
```bash
GET /
```

### List Required Features
```bash
GET /features
```

### Single District Prediction
```bash
POST /predict
Content-Type: application/json

{
  "t_tl": 150000,
  "working_age_pct": 0.55,
  "under5_pct": 0.18,
  "district_sex_ratio": 0.98,
  "log_area_km2": 7.5,
  "log_pop_density": 5.2,
  "compactness": 0.65,
  "abs_lat": 1.5,
  "facility_count": 25,
  "facility_density": 0.0002,
  "log_facility_per_100k": 2.8,
  "is_urban_survey_cluster": 0,
  "region_encoded": 3,
  "district_name": "Kampala"
}
```

**Response:**
```json
{
  "district_name": "Kampala",
  "risk_score": 0.8234,
  "risk_tier": "High",
  "iso_score": 0.8456,
  "kde_score": 0.7891
}
```

### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

[
  { /* district 1 data */ },
  { /* district 2 data */ },
  ...
]
```

Maximum 200 districts per batch.

## 📊 Input Features

| Feature | Description | Example |
|---------|-------------|---------|
| `t_tl` | Total population | 150000 |
| `working_age_pct` | Working-age population % | 0.55 |
| `under5_pct` | Under-5 population % | 0.18 |
| `district_sex_ratio` | Sex ratio (male/female) | 0.98 |
| `log_area_km2` | Log of district area (km²) | 7.5 |
| `log_pop_density` | Log of population density | 5.2 |
| `compactness` | Shape compactness index | 0.65 |
| `abs_lat` | Absolute latitude | 1.5 |
| `facility_count` | Number of health facilities | 25 |
| `facility_density` | Health facility density | 0.0002 |
| `log_facility_per_100k` | Log facilities per 100k pop | 2.8 |
| `is_urban_survey_cluster` | Urban cluster flag (0 or 1) | 0 |
| `region_encoded` | Encoded region identifier | 3 |

## 🎨 Risk Tiers

- **🟢 Low Risk:** Score 0.00 - 0.45 (Normal district)
- **🟡 Medium Risk:** Score 0.45 - 0.75 (Somewhat unusual)
- **🔴 High Risk:** Score 0.75 - 1.00 (Outlier - needs attention)

## 🌐 Live Deployment

**API:** [Your Render URL will be here]

**Dashboard:** [Your Render Dashboard URL will be here]

## 📁 Project Structure

```
MACHINE-LEARNING-SW-ML-38/
├── models/
│   └── best_model.pkl              # Trained Isolation Forest model
├── deployment/
│   ├── app.py                      # FastAPI application
│   ├── dashboard.py                # Streamlit dashboard
│   └── requirements_deploy.txt     # Deployment dependencies
├── requirements_render.txt         # All dependencies for Render
├── README.md                       # This file
└── WHY_ISOLATION_FOREST_VISUAL.md  # Model explanation
```

## 🔧 Technology Stack

- **API Framework:** FastAPI
- **ML Library:** scikit-learn
- **Model:** Isolation Forest
- **Dashboard:** Streamlit
- **Deployment:** Render
- **Language:** Python 3.11+

## 📖 Documentation

- [Why Isolation Forest?](WHY_ISOLATION_FOREST_VISUAL.md) - Model selection explanation
- [Model Comparison](MODEL_COMPARISON_SUMMARY.md) - Performance comparison of 7 models
- [Deployment Guide](GITHUB_RENDER_DEPLOYMENT_GUIDE.md) - Step-by-step deployment instructions

## 🎯 Use Cases

### 1. Resource Allocation
Identify top 10 highest-risk districts to prioritize TB testing kit distribution.

### 2. Intervention Planning
Find districts with high risk + low facility count to plan new health facility construction.

### 3. Monitoring Trends
Track risk score changes over time to detect emerging TB hotspots.

### 4. Regional Comparison
Compare TB risk across Uganda's 4 regions (Central, Eastern, Northern, Western).

## 🔐 Security

- HTTPS enabled by default on Render
- Input validation via Pydantic models
- Rate limiting recommended for production
- API authentication recommended for production

## 📊 Model Details

### Why Isolation Forest?

Isolation Forest was selected as the best model after evaluating 7 different algorithms:

1. **Isolation Forest** - 0.701 AUC-ROC ✅ (Winner)
2. Ensemble (IsoF+KDE) - 0.687 AUC-ROC
3. Spectral Clustering - 0.673 AUC-ROC
4. KDE - 0.675 AUC-ROC
5. Gaussian Mixture - 0.644 AUC-ROC
6. HDBSCAN - 0.520 AUC-ROC
7. DBSCAN - 0.500 AUC-ROC

**Key advantages:**
- Designed for anomaly detection (finding high-risk outliers)
- Works with unlabeled data (unsupervised)
- Fast predictions (47 ms for 135 districts)
- Interpretable results
- Handles small datasets well

## 🤝 Contributing

This is a public health project for Uganda's Ministry of Health. Contributions are welcome!

## 📄 License

MIT License - See LICENSE file for details

## 👥 Authors

- Michael Kakooza - [@KAKOOZAMICHAEL](https://github.com/KAKOOZAMICHAEL)

## 📞 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- Uganda Ministry of Health
- WHO TB data
- HDX humanitarian data
- OpenStreetMap health facility data

---

**Built with ❤️ for better TB prevention in Uganda**
