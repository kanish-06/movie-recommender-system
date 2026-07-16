import streamlit as st
import pickle
import pandas as pd

movies_dict = pickle.load(open('models/movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('models/similarity.pkl', 'rb'))

def recommend(movie):
    movie_index = movies[movies['title']==movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    recommended_movies = []
    for i in movie_list:
        movie_id = i[0]
        # fetching poster using API is tbd
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

st.title('Movie Recommender System')

selected_movie = st.selectbox("How would you like to predict?",movies['title'].values)

button = st.button("Recommend")

if button:
    recommendations = recommend(selected_movie)
    for i in recommendations:
        st.write(i)