import streamlit as st
import pandas as pd

st.set_page_config(page_title="Book Recommender", layout="wide")

st.title("📚 Book Recommendation System")

books = pd.read_csv("books_small.csv")

selected_book = st.selectbox("Choose a book", books["title"].values)

if st.button("Recommend"):
    recs = books.sample(5)  # replace with your model later
    cols = st.columns(5)

    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i]:
            if "image_url" in row:
                st.image(row["image_url"], use_container_width=True)
            st.caption(row["title"])
