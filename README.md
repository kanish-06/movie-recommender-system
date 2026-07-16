#  Movie Recommendation System

A content-based movie recommendation system built using **Python**, **Scikit-learn**, and **Streamlit** that recommends similar movies based on their metadata.

> This project was built as part of my learning journey to understand content-based recommendation systems. I initially followed a tutorial to learn the implementation and plan to extend it with additional features and improvements

## Features

- Search for a movie
- Get recommendations for similar movies
- Interactive Streamlit web interface
- Fast recommendation retrieval using precomputed similarity scores

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Dataset

- TMDB Movie Dataset
## Project Structure

```text
movie-recommender-system/
├── app.py                     # Streamlit application
├── generate_model.py          # generates model artifacts from raw data
├── README.md                  # documentation
├── requirements.txt           # dependencies
├── .gitignore  
│
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── models/
│   ├── movie_dict.pkl
│   └── similarity.pkl         # Generated locally (not tracked by Git)
│
└── notebooks/
    └── exploration.ipynb
```

## Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/<your-username>/movie-recommender-system.git
cd movie-recommender-system

pip install -r requirements.txt
```

### Generate Model Files

The `similarity.pkl` file is **not included** in this repository because it exceeds GitHub's 100 MB file size limit.

Generate the required model files by running:

```bash
python generate_model.py
```

This will create:

```
models/
├── movie_dict.pkl
└── similarity.pkl
```

Once the model files have been generated, start the application:

```bash
streamlit run app.py
```

## Future Improvements

- [ ] Genre-based filtering
- [ ] Recommendation explanations
- [ ] Deploy the application
