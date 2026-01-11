
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
from config import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME


st.set_page_config(page_title="Sakila ML Dashboard", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction"])

# --- MySQL Connection Helper ---

# Use SQLAlchemy engine for pandas compatibility
@st.cache_resource
def get_sqlalchemy_engine():
    return create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")


# Helper: Query DB and return DataFrame
def run_query(query, params=None):
    engine = get_sqlalchemy_engine()
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    return df


if page == "Home":
    st.title("Sakila Movie Rentals Dashboard")
    st.image("https://www.visitberlin.de/system/files/styles/visitberlin_teaser_search_visitberlin_mobile_1x/private/image/delphiLUX_Saal4_c_YorckKinogruppe_AdrianSchulz_web.jpg.jpg?itok=eTtyhBmF", width=600)
    st.write("Welcome to the Sakila Movie Rentals Dashboard! Use the sidebar to navigate between pages.")

elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    # --- Daily Rentals by Store in 2005 ---
    st.subheader("Daily Rentals by Store in 2005")
    rentals_query = '''
        SELECT DATE(r.rental_date) AS rental_day, i.store_id, COUNT(*) AS rentals
        FROM rental r
        JOIN inventory i ON r.inventory_id = i.inventory_id
        WHERE YEAR(r.rental_date) = 2005
        GROUP BY rental_day, i.store_id
        ORDER BY rental_day, i.store_id
    '''
    rentals_df = run_query(rentals_query)
    rentals_pivot = rentals_df.pivot(index="rental_day", columns="store_id", values="rentals").fillna(0)
    st.line_chart(rentals_pivot)

    # --- Total Benefit by Store ---
    st.subheader("Total Benefit by Store")
    benefit_query = '''
        SELECT i.store_id, SUM(p.amount) AS total_benefit
        FROM payment p
        JOIN rental r ON p.rental_id = r.rental_id
        JOIN inventory i ON r.inventory_id = i.inventory_id
        GROUP BY i.store_id
    '''
    benefit_df = run_query(benefit_query)
    benefit_df = benefit_df.set_index("store_id")
    st.bar_chart(benefit_df)

    # --- Top 5 Most Rented Movies by Store in 2005 ---
    st.subheader("Top 5 Most Rented Movies by Store in 2005")
    top5_query = '''
        SELECT i.store_id, f.title, COUNT(*) AS rentals
        FROM rental r
        JOIN inventory i ON r.inventory_id = i.inventory_id
        JOIN film f ON i.film_id = f.film_id
        WHERE YEAR(r.rental_date) = 2005
        GROUP BY i.store_id, f.title
        ORDER BY i.store_id, rentals DESC
    '''
    top5_df = run_query(top5_query)
    # Get top 5 per store
    top5_df = top5_df.groupby('store_id').apply(lambda x: x.nlargest(5, 'rentals')).reset_index(drop=True)
    st.dataframe(top5_df)


elif page == "Prediction":
    st.title("Movie Description Similarity Search")
    st.write("Enter a movie description to find the top 3 most similar movies by description.")
    user_input = st.text_area("Movie Description", "")
    if st.button("Get Your Prediction"):
        # Get all movies from Sakila
        movie_query = '''
            SELECT f.title, f.description, f.rating, c.name AS genre
            FROM film f
            LEFT JOIN film_category fc ON f.film_id = fc.film_id
            LEFT JOIN category c ON fc.category_id = c.category_id
            WHERE f.description IS NOT NULL AND f.description != ''
        '''
        movie_data = run_query(movie_query)
        # If no movies, show error
        if movie_data.empty:
            st.error("No movie descriptions found in the database.")
        else:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            movie_embs = model.encode(movie_data['description'].tolist())
            user_emb = model.encode([user_input])[0]
            sims = cosine_similarity([user_emb], movie_embs)[0]
            top_idx = np.argsort(sims)[::-1][:3]
            st.subheader("Top 3 Most Similar Movies:")
            for idx in top_idx:
                st.write(f"**Title:** {movie_data['title'].iloc[idx]} | **Genre:** {movie_data['genre'].iloc[idx]} | **Rating:** {movie_data['rating'].iloc[idx]} | **Similarity:** {sims[idx]:.2f}")
                st.write(f"**Description:** {movie_data['description'].iloc[idx]}")

