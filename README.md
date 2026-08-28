# 🧳 Tourism Experience Analytics

A machine learning platform that analyzes tourism data to predict attraction ratings, classify visitor modes, and recommend personalized attractions — helping travel platforms and tourism agencies enhance user experience through data-driven insights.

## 🔍 Problem Statement

Tourism agencies and travel platforms aim to enhance user experiences by leveraging data to provide personalized recommendations, predict user satisfaction, and classify potential user behavior. This project analyzes user preferences, travel patterns, and attraction features to achieve three primary objectives: regression, classification, and recommendation.

## 🚀 Live App

(https://tourism-project-lyvtxyzuvo8cxwpljc2ce2.streamlit.app/)
## 📊 Dataset

The dataset consists of 9 linked tables covering:
- **Transaction data** — user visits, ratings, visit mode, year/month of visit
- **User data** — geographic demographics (continent, region, country, city)
- **Attraction data** — attraction type, location, address
- **Reference tables** — City, Country, Region, Continent, Visit Mode, Attraction Type

After merging, the master dataset contains **52,922 cleaned transaction records** across 22 columns.

## 🎯 Objectives

**1. Regression — Predicting Attraction Ratings**
Predicts the rating (1-5) a user might give to an attraction based on demographics, visit details, and attraction features.

**2. Classification — Visit Mode Prediction**
Predicts the mode of visit (Business, Family, Couples, Friends, Solo) based on user and attraction data.

**3. Recommendation — Personalized Attraction Suggestions**
Item-based collaborative filtering recommends attractions based on similarity to a user's previously visited and rated attractions.

## 🧠 Model Results

**Regression (Rating Prediction)**
| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 0.503 | 0.734 |
| Random Forest | 0.518 | 0.717 |
| **XGBoost (best)** | **0.501** | **0.736** |

**Classification (Visit Mode Prediction)**
| Model | Accuracy |
|---|---|
| Random Forest (base features) | 51.3% |
| XGBoost (base features) | 49.9% |
| **Random Forest + user history feature (best)** | **91.7%** |

**Recommendation System**
Item-based collaborative filtering using cosine similarity on a user-attraction rating matrix (33,526 users × 30 attractions).

## 🛠️ Tech Stack

- **Language:** Python
- **ML/Data:** scikit-learn, XGBoost, pandas, numpy
- **Web App:** Streamlit
- **Deployment:** Streamlit Cloud

## 📁 Project Structure

```
Tourism-Experience-Analytics/
├── app.py                          # Streamlit web application
├── Tourism_0X...ipynb               # Data merging, cleaning, EDA, feature engineering, model training
├── rating_model.pkl                 # Trained XGBoost regression model
├── rating_feature_cols.pkl          # Feature columns for rating model
├── visitmode_model.pkl              # Trained Random Forest classification model
├── visitmode_feature_cols.pkl       # Feature columns for visit mode model
├── user_common_mode_encoder.pkl     # Label encoder for user's historical common visit mode
├── user_mode_lookup.pkl             # Per-user historical common visit mode lookup table
├── item_similarity.pkl              # Attraction-to-attraction similarity matrix
├── user_item_matrix.pkl             # User-attraction rating matrix
├── tourism_cat_encoders.pkl         # Label encoders for categorical features
├── item_lookup.pkl                  # Attraction ID to name/type lookup
├── master_tourism_fe.csv            # Cleaned and feature-engineered dataset
└── README.md
```

## ⚙️ How It Works

1. **Data Cleaning & Merging** — joined 9 linked tables into one master dataset, handled missing values and inconsistencies
2. **EDA** — analyzed user distribution by continent/country, attraction popularity, rating patterns, and visit mode trends
3. **Feature Engineering** — user-level and attraction-level aggregated features (average ratings, visit counts, popularity), categorical encoding
4. **Model Training** — trained regression models (rating prediction) and classification models (visit mode prediction)
5. **Recommendation System** — built an item-based collaborative filtering engine using cosine similarity
6. **Streamlit App** — multi-page interface for predictions, recommendations, and data insights

## ▶️ Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📈 App Features

- **Predict Rating** — enter visit and attraction details to get a predicted rating
- **Predict Visit Mode** — predict whether a visit is Business, Family, Couples, Friends, or Solo
- **Recommendations** — enter a User ID to get personalized attraction suggestions
- **Data Insights** — explore dataset distributions and trends
- **About** — project and tech stack overview

## 👩‍💻 Author

Disha Garg
