import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Tourism Experience Analytics", layout="wide", page_icon="🧳")

# ---------------------------------------------------------------
# Styling
# ---------------------------------------------------------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #ff7e5f 0%, #feb47b 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.main-header h1 { color: white; margin: 0; }
.main-header p { color: #fff0e6; margin: 0.3rem 0 0 0; }

.result-card {
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    background: linear-gradient(135deg, #11998e, #38ef7d);
}
.result-card h2 { color: white; margin: 0; }
.result-card p { color: white; font-size: 0.9rem; margin: 0.3rem 0 0 0; }

.mode-card {
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    background: linear-gradient(135deg, #4e54c8, #8f94fb);
}
.mode-card h2 { color: white; margin: 0; }
.mode-card p { color: white; font-size: 0.9rem; margin: 0.3rem 0 0 0; }

.attraction-card {
    background: #f8f9fb;
    padding: 1rem 1.2rem;
    border-radius: 10px;
    margin-bottom: 0.6rem;
    border-left: 4px solid #ff7e5f;
}
div[data-testid="stMarkdownContainer"] .attraction-card,
div[data-testid="stMarkdownContainer"] .attraction-card * {
    color: #1e1e1e !important;
}
.attraction-card b { color: #ff7e5f !important; }

[data-testid="stMetric"] {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 12px;
}
[data-testid="stMetric"] label, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #1e1e1e !important;
}
[data-testid="stMetricValue"] {
    color: #ff7e5f !important;
    font-weight: 700 !important;
}

div[data-testid="stMarkdownContainer"] .section-card {
    background: #f8f9fb;
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}
div[data-testid="stMarkdownContainer"] .section-card * {
    color: #1e1e1e !important;
}
div[data-testid="stMarkdownContainer"] .section-card b {
    color: #ff7e5f !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("rating_model.pkl", "rb") as f:
        rating_model = pickle.load(f)
    with open("rating_feature_cols.pkl", "rb") as f:
        rating_feature_cols = pickle.load(f)
    with open("visitmode_model.pkl", "rb") as f:
        visitmode_model = pickle.load(f)
    with open("visitmode_feature_cols.pkl", "rb") as f:
        visitmode_feature_cols = pickle.load(f)
    with open("user_common_mode_encoder.pkl", "rb") as f:
        user_common_mode_encoder = pickle.load(f)
    with open("user_mode_lookup.pkl", "rb") as f:
        user_mode_lookup = pickle.load(f)
    with open("item_similarity.pkl", "rb") as f:
        item_similarity_df = pickle.load(f)
    with open("user_item_matrix.pkl", "rb") as f:
        user_item_matrix = pickle.load(f)
    with open("tourism_cat_encoders.pkl", "rb") as f:
        cat_encoders = pickle.load(f)
    with open("item_lookup.pkl", "rb") as f:
        item_lookup = pickle.load(f)
    return (rating_model, rating_feature_cols, visitmode_model, visitmode_feature_cols,
            user_common_mode_encoder, user_mode_lookup, item_similarity_df,
            user_item_matrix, cat_encoders, item_lookup)


(rating_model, rating_feature_cols, visitmode_model, visitmode_feature_cols,
 user_common_mode_encoder, user_mode_lookup, item_similarity_df,
 user_item_matrix, cat_encoders, item_lookup) = load_artifacts()


@st.cache_data
def load_master_data():
    try:
        return pd.read_csv("master_tourism_fe.csv")
    except FileNotFoundError:
        return None


def recommend_attractions(user_id, n=5):
    if user_id not in user_item_matrix.index:
        return None
    user_ratings = user_item_matrix.loc[user_id].dropna()
    if len(user_ratings) == 0:
        return None
    scores = pd.Series(dtype=float)
    for attraction_id, rating in user_ratings.items():
        similar_scores = item_similarity_df[attraction_id] * rating
        scores = scores.add(similar_scores, fill_value=0)
    scores = scores.drop(user_ratings.index, errors="ignore")
    scores = scores.sort_values(ascending=False)
    top_ids = scores.head(n).index.tolist()
    return item_lookup[item_lookup["AttractionId"].isin(top_ids)].drop_duplicates("AttractionId")


VISIT_MODE_EMOJI = {
    "Business": "💼", "Family": "👨‍👩‍👧", "Couples": "💑",
    "Friends": "🧑‍🤝‍🧑", "Solo": "🧍",
}

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div class="main-header">
<h1>🧳 Tourism Experience Analytics</h1>
<p>Predict attraction ratings, visit modes, and get personalized recommendations</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", ["Predict Rating", "Predict Visit Mode", "Recommendations", "Data Insights", "About"])

# -----------------------------------------------------------
# PREDICT RATING PAGE
# -----------------------------------------------------------
if page == "Predict Rating":
    st.header("⭐ Predict Attraction Rating")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Visit Details**")
        visit_year = st.number_input("Visit Year", 2015, 2026, 2024)
        visit_month = st.selectbox("Visit Month", list(range(1, 13)))
        visit_mode = st.selectbox("Visit Mode", cat_encoders["VisitModeText"].classes_.tolist())
        continent = st.selectbox("Continent", cat_encoders["Continent"].classes_.tolist())
    with col2:
        st.markdown("**Attraction & User Profile**")
        country = st.selectbox("Country", cat_encoders["Country"].classes_.tolist())
        attraction_type = st.selectbox("Attraction Type", cat_encoders["AttractionType"].classes_.tolist())
        user_avg_rating = st.slider("User's Average Past Rating", 1.0, 5.0, 4.0)
        user_total_visits = st.number_input("User's Total Past Visits", 1, 60, 3)

    st.markdown("**Attraction Stats**")
    c1, c2 = st.columns(2)
    with c1:
        attraction_avg_rating = st.slider("Attraction's Average Rating (historical)", 1.0, 5.0, 4.2)
    with c2:
        attraction_popularity = st.number_input("Attraction Popularity (visit count)", 1, 15000, 3000)

    if st.button("🔍 Predict Rating", use_container_width=True):
        input_dict = {
            "VisitYear": visit_year, "VisitMonth": visit_month,
            "VisitModeText_enc": cat_encoders["VisitModeText"].transform([visit_mode])[0],
            "Continent_enc": cat_encoders["Continent"].transform([continent])[0],
            "Country_enc": cat_encoders["Country"].transform([country])[0],
            "AttractionType_enc": cat_encoders["AttractionType"].transform([attraction_type])[0],
            "user_avg_rating": user_avg_rating, "user_total_visits": user_total_visits,
            "attraction_avg_rating": attraction_avg_rating, "attraction_popularity": attraction_popularity,
        }
        input_df = pd.DataFrame([input_dict])[rating_feature_cols]
        pred_rating = rating_model.predict(input_df)[0]
        pred_rating = float(np.clip(pred_rating, 1, 5))
        stars = "⭐" * round(pred_rating)

        st.subheader("Prediction Result")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"""
            <div class="result-card">
                <h2>{pred_rating:.2f} / 5</h2>
                <p>{stars}</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.metric("Predicted Rating", f"{pred_rating:.2f}")
            st.progress(pred_rating / 5)

# -----------------------------------------------------------
# PREDICT VISIT MODE PAGE
# -----------------------------------------------------------
elif page == "Predict Visit Mode":
    st.header("🧭 Predict Visit Mode")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Visit Details**")
        visit_year = st.number_input("Visit Year", 2015, 2026, 2024, key="vm_year")
        visit_month = st.selectbox("Visit Month", list(range(1, 13)), key="vm_month")
        continent = st.selectbox("Continent", cat_encoders["Continent"].classes_.tolist(), key="vm_cont")
        country = st.selectbox("Country", cat_encoders["Country"].classes_.tolist(), key="vm_country")
        attraction_type = st.selectbox("Attraction Type", cat_encoders["AttractionType"].classes_.tolist(), key="vm_type")
    with col2:
        st.markdown("**Rating & History**")
        rating = st.slider("Rating Given", 1, 5, 4, key="vm_rating")
        user_avg_rating = st.slider("User's Average Past Rating", 1.0, 5.0, 4.0, key="vm_uavg")
        user_total_visits = st.number_input("User's Total Past Visits", 1, 60, 3, key="vm_visits")
        attraction_avg_rating = st.slider("Attraction's Average Rating", 1.0, 5.0, 4.2, key="vm_aavg")
        attraction_popularity = st.number_input("Attraction Popularity", 1, 15000, 3000, key="vm_pop")

    user_id_lookup = st.number_input("User ID (optional, for historical pattern)", min_value=0, value=0, key="vm_uid")

    if st.button("🔍 Predict Visit Mode", use_container_width=True):
        match = user_mode_lookup[user_mode_lookup["UserId"] == user_id_lookup]
        if len(match) > 0:
            common_mode = match["user_common_mode"].iloc[0]
        else:
            common_mode = user_mode_lookup["user_common_mode"].mode()[0]
        common_mode_enc = user_common_mode_encoder.transform([common_mode])[0]

        input_dict = {
            "VisitYear": visit_year, "VisitMonth": visit_month,
            "Continent_enc": cat_encoders["Continent"].transform([continent])[0],
            "Country_enc": cat_encoders["Country"].transform([country])[0],
            "AttractionType_enc": cat_encoders["AttractionType"].transform([attraction_type])[0],
            "Rating": rating, "user_avg_rating": user_avg_rating,
            "user_total_visits": user_total_visits,
            "attraction_avg_rating": attraction_avg_rating,
            "attraction_popularity": attraction_popularity,
            "user_common_mode_enc": common_mode_enc,
        }
        input_df = pd.DataFrame([input_dict])[visitmode_feature_cols]
        pred_mode = visitmode_model.predict(input_df)[0]
        emoji = VISIT_MODE_EMOJI.get(pred_mode, "🧳")

        st.subheader("Prediction Result")
        st.markdown(f"""
        <div class="mode-card">
            <h2>{emoji} {pred_mode}</h2>
            <p>Predicted Visit Mode</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------
# RECOMMENDATIONS PAGE
# -----------------------------------------------------------
elif page == "Recommendations":
    st.header("🎯 Personalized Attraction Recommendations")
    st.write("Enter a User ID to get attraction suggestions based on similar travelers' preferences.")

    user_id = st.number_input("User ID", min_value=int(user_item_matrix.index.min()),
                                max_value=int(user_item_matrix.index.max()),
                                value=int(user_item_matrix.index[0]))

    if st.button("🎯 Get Recommendations", use_container_width=True):
        results = recommend_attractions(user_id, n=5)
        if results is None or len(results) == 0:
            st.warning("No recommendations available for this User ID — they may have no visit history.")
        else:
            st.subheader(f"Top Recommendations for User {user_id}")
            for _, row in results.iterrows():
                st.markdown(f"""
                <div class="attraction-card">
                    <b>{row['Attraction']}</b><br>{row['AttractionType']}
                </div>
                """, unsafe_allow_html=True)

# -----------------------------------------------------------
# DATA INSIGHTS PAGE
# -----------------------------------------------------------
elif page == "Data Insights":
    st.header("📊 Dataset Insights")
    df = load_master_data()

    if df is None:
        st.warning("master_tourism_fe.csv not found in the app folder.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Transactions by Continent**")
            st.bar_chart(df["Continent"].value_counts())
        with c2:
            st.markdown("**Visit Mode Distribution**")
            st.bar_chart(df["VisitModeText"].value_counts())

        st.markdown("**Top 10 Attraction Types by Visit Count**")
        st.bar_chart(df["AttractionType"].value_counts().head(10))

        st.markdown("**Average Rating by Attraction Type**")
        st.bar_chart(df.groupby("AttractionType")["Rating"].mean().sort_values(ascending=False).head(10))

        st.caption(f"Insights generated from {len(df):,} transaction records.")

# -----------------------------------------------------------
# ABOUT PAGE
# -----------------------------------------------------------
else:
    st.header("ℹ️ About This Project")
    st.markdown("""
    <div class="section-card">
    <b>Tourism Experience Analytics</b> combines three machine learning tasks:

    - <b>Regression</b> — predicts the rating a user might give to an attraction<br>
    - <b>Classification</b> — predicts a user's likely visit mode (Business, Family, Couples, Friends, Solo)<br>
    - <b>Recommendation</b> — suggests attractions using item-based collaborative filtering

    <br><br>Built with Python, scikit-learn, XGBoost, and Streamlit, using a merged dataset of
    user transactions, attractions, and geography spanning multiple linked tables.
    </div>
    """, unsafe_allow_html=True)