import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load Trained SVD Model
# -------------------------------
model = pickle.load(open("svd_model.pkl", "rb"))

# -------------------------------
# Load Dataset Files
# -------------------------------
books = pd.read_csv("books.csv")
ratings = pd.read_csv("ratings.csv")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📚 Book Recommendation System")
st.write("Get personalized book recommendations using SVD Model")

st.subheader("🔎 Enter User ID")

user_id = st.number_input("Enter User ID:", min_value=1, step=1)

if st.button("Recommend Books"):

    # Check if user exists
    if user_id not in ratings["user_id"].unique():
        st.error("❌ User ID not found in dataset")
    
    else:
        st.success("Generating Recommendations...")

        # Books already rated by user
        rated_books = ratings[ratings["user_id"] == user_id]["book_id"].tolist()

        # Books NOT rated by user
        not_rated = books[~books["book_id"].isin(rated_books)]

        predictions = []

        # Predict ratings for unseen books
        for book_id in not_rated["book_id"]:
            pred = model.predict(user_id, book_id)
            predictions.append((book_id, pred.est))

        # Convert to DataFrame
        pred_df = pd.DataFrame(predictions, columns=["book_id", "predicted_rating"])

        # Sort by highest predicted rating
        pred_df = pred_df.sort_values("predicted_rating", ascending=False)

        # Get Top 5
        top_books = pred_df.head(5)

        # Merge with book details
        recommended = top_books.merge(books, on="book_id")

        st.subheader("🎯 Top 5 Recommended Books For You")

        for _, row in recommended.iterrows():
            st.write(f"📖 **{row['title']}** by {row['authors']}")
