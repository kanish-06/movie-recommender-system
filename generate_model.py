"""
Generate the model artifacts for the Movie Recommendation System.

This script:
1. Loads the TMDB datasets.
2. Preprocesses movie metadata.
3. Creates content-based tags.
4. Computes the cosine similarity matrix.
5. Saves:
   - models/movie_dict.pkl
   - models/similarity.pkl
"""

import ast
import pickle
from pathlib import Path

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

MOVIES_CSV = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = DATA_DIR / "tmdb_5000_credits.csv"

MOVIE_DICT_PATH = MODELS_DIR / "movie_dict.pkl"
SIMILARITY_PATH = MODELS_DIR / "similarity.pkl"

_stemmer = PorterStemmer()

def load_data(movies_csv: Path = MOVIES_CSV, credits_csv: Path = CREDITS_CSV) -> pd.DataFrame:
    print("Loading datasets...")

    movies = pd.read_csv(movies_csv)
    credits = pd.read_csv(credits_csv)

    movies = movies.merge(credits, on="title")

    return movies



def _convert(obj: str) -> list:
    return [i["name"] for i in ast.literal_eval(obj)]


def _convert_cast(obj: str) -> list:
    names = []
    parsed = ast.literal_eval(obj)
    counter = 0
    for i in parsed:
        if counter == 3:
            break
        names.append(i["name"])
        counter += 1
    return names


def _convert_crew(obj: str):
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            return [i["name"]]


def _stem(text: str) -> str:
    return " ".join(_stemmer.stem(word) for word in text.split())


def preprocess_data(movies: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning data...")

    movies = movies[["genres", "id", "keywords", "title", "overview", "cast", "crew"]]

    movies = movies.dropna()

    movies["genres"] = movies["genres"].apply(_convert)
    movies["keywords"] = movies["keywords"].apply(_convert)

    movies["cast"] = movies["cast"].apply(_convert_cast)

    movies["crew"] = movies["crew"].apply(_convert_crew)

    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])

    movies = movies.dropna()

    return movies


def create_tags(movies: pd.DataFrame) -> pd.DataFrame:
    print("Creating tags...")

    ## Combine all relevant text into a single feature for each movie
    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    new_df = movies[["id", "title", "tags"]].copy()

    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))

    new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

    new_df["tags"] = new_df["tags"].apply(_stem)

    return new_df

def vectorize_movies(new_df):
    print("Vectorizing...")

    vectors=CountVectorizer(max_features=5000,stop_words="english").fit_transform(new_df["tags"]).toarray()

    return vectors


def compute_similarity(vectors):
    print("Computing similarity matrix...")

    return cosine_similarity(vectors)

def save_models(new_df: pd.DataFrame, similarity, models_dir: Path = MODELS_DIR) -> None:
    print("Saving model...")

    models_dir.mkdir(parents=True, exist_ok=True)

    with open(MOVIE_DICT_PATH, "wb") as file:
        pickle.dump(new_df.to_dict(), file)

    with open(SIMILARITY_PATH, "wb") as file:
        pickle.dump(similarity, file)

    print("Models saved successfully!")


def main() -> None:
    movies = load_data()
    movies = preprocess_data(movies)
    new_df = create_tags(movies)
    vectors = vectorize_movies(new_df)
    similarity = compute_similarity(vectors)
    save_models(new_df, similarity)

    print("Done!")


if __name__ == "__main__":
    main()