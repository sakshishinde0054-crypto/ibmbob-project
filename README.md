# 🍽️ Zomato Online Order Prediction

A full-stack Machine Learning web application that predicts whether a restaurant
accepts **online orders** based on its rating, votes, cost, table booking, and
restaurant type — built entirely in Python.

---

## 📁 Project Structure

```
zomato_project/
│
├── train_model.py          # Data preprocessing + ML model training
├── app.py                  # Flask REST API backend
├── streamlit_app.py        # Streamlit Python frontend
├── requirements.txt        # All Python dependencies
├── README.md               # This file
│
├── models/                 # Auto-created after training
│   ├── model.pkl           # Trained Random Forest pipeline
│   ├── label_encoders.pkl  # Fitted label encoders
│   ├── feature_cols.pkl    # Ordered feature list
│   ├── metrics.json        # Model evaluation metrics
│   └── encoder_classes.json# Dropdown values for UI
│
└── static/                 # Auto-created after training
    ├── confusion_matrix.png
    └── feature_importances.png
```

---

## 📊 Dataset

| Field | Description |
|---|---|
| `name` | Restaurant name |
| `online_order` | **Target** — Yes / No |
| `book_table` | Table booking available — Yes / No |
| `rate` | Rating out of 5 (e.g. `4.1/5`) |
| `votes` | Number of customer votes |
| `approx_cost(for two people)` | Approximate cost for 2 people (₹) |
| `listed_in(type)` | Restaurant type (Buffet, Cafes, etc.) |

- **Source:** Zomato Bangalore restaurant dataset
- **Size:** 148 restaurants
- **Task:** Binary Classification — predict `online_order` (Yes=1 / No=0)

---

## 🤖 ML Model

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Features | rate, votes, cost, book_table, listed_in_type |
| Target | online_order (binary) |
| Train/Test Split | 80% / 20% |
| Cross-Validation | 5-fold |
| Class Weighting | Balanced |

---

## 🚀 Getting Started

### 1. Clone / Download the project

```bash
cd zomato_project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
# Uses default dataset path
python train_model.py

# Or specify custom path
python train_model.py --data "C:/path/to/Zomato-data-.csv"
```

Expected output:
```
[1/5] Loading dataset...
[2/5] Feature matrix : (118, 5)
[3/5] Training Random Forest...
[4/5] Metrics:
      accuracy                 : 0.8333
      precision                : 0.8500
      recall                   : 0.8700
      f1_score                 : 0.8598
      roc_auc                  : 0.8900
      cv_accuracy_5fold        : 0.8200
[5/5] Saving artefacts...
✅  Training complete — artefacts saved to ./models/
```

### 5. Start the Flask backend (Terminal 1)

```bash
python app.py
```

API running at: `http://localhost:5000`

### 6. Start the Streamlit frontend (Terminal 2)

```bash
streamlit run streamlit_app.py
```

App running at: `http://localhost:8501`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Model evaluation metrics |
| `GET` | `/meta` | Encoder classes + metrics |
| `GET` | `/feature-importance` | Feature importance scores |
| `POST` | `/predict` | Single or batch prediction |

### Example — Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rate": 4.1,
    "votes": 775,
    "cost": 800,
    "book_table": "Yes",
    "listed_in_type": "Buffet"
  }'
```

Response:
```json
{
  "prediction": 1,
  "label": "Accepts Online Orders",
  "probability_online": 0.87,
  "probability_no_online": 0.13
}
```

---

## 🖥️ Streamlit Frontend Pages

| Page | Description |
|---|---|
| 🏠 Home | Project overview and dataset summary |
| 🔮 Predict | Interactive form — predict online order for a restaurant |
| 📊 EDA | Exploratory charts — rating, cost, votes distributions |
| 📈 Model Performance | Accuracy, F1, ROC-AUC, confusion matrix |
| 📋 Dataset | Browse the raw Zomato dataset |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn (Random Forest) |
| Visualisation | matplotlib, seaborn, plotly |
| Backend API | Flask + flask-cors |
| Frontend | Streamlit + streamlit-option-menu |
| HTTP Client | requests |

---

## 📌 Notes

- Run `train_model.py` **before** starting `app.py` or `streamlit_app.py`
- The `models/` and `static/` folders are auto-created during training
- For production deployment, replace `flask dev server` with `gunicorn`

```bash
# Production backend
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## 👤 Author

Built with Python · scikit-learn · Flask · Streamlit
