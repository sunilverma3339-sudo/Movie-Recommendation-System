# Movie Recommendation System

## Overview
The Movie Recommendation System is a Python-based application that recommends movies using content similarity, collaborative filtering, and a hybrid ranking strategy. It also supports genre prediction from plot descriptions and an interactive web interface.

## Features
- Content-based recommendation using TF-IDF and cosine similarity
- Collaborative filtering based on user ratings
- Hybrid recommendations combining content similarity, ratings, and collaborations
- Movie search by title, genre, or keywords
- Genre prediction from plot descriptions
- CLI experience and web UI with forms for search, recommendations, rating, and genre prediction

## Installation
1. Install Python 3.9+
2. Create a virtual environment (recommended)
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
### CLI
Run the CLI tool:

```bash
python app.py
```

The CLI supports:
- Search movies
- Get content-based recommendations
- Get collaborative recommendations
- Get hybrid recommendations
- Predict genres from plot text
- Save user ratings for collaborative filtering

### Web UI
Start the web application:

```bash
python app.py --web
```

Then visit:

```
http://127.0.0.1:5000
```

## Data
- Place the movie dataset at `data/movies.csv`
- Ratings are stored in `data/user_ratings.csv` once submitted through the app or CLI

## Notes
- The `movie_recommender.py` module provides a shared `MovieRecommender` class for reuse in scripts or web interfaces.
- The web UI is intentionally simple and built with Flask so it can be extended later.
