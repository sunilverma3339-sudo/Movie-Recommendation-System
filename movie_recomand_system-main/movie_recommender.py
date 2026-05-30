import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies.csv")
USER_RATINGS_PATH = os.path.join(BASE_DIR, "data", "user_ratings.csv")


def _clean_text_column(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.strip('"')
        .str.replace(r"\s+", " ", regex=True)
    )


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = _clean_text_column(df[col])

    if "GENRE" in df.columns:
        df["GENRE"] = _clean_text_column(df["GENRE"]).str.replace(r"\n", "", regex=True)
    if "ONE-LINE" in df.columns:
        df["ONE-LINE"] = _clean_text_column(df["ONE-LINE"]).str.replace(r"\n", " ", regex=True)

    df = df[df["ONE-LINE"].notna() & (df["ONE-LINE"] != "nan")]
    df = df[df["GENRE"].notna() & (df["GENRE"] != "nan")]

    df["MOVIES"] = df["MOVIES"].astype(str).str.strip().str.strip('"')
    df["GENRE_LIST"] = df["GENRE"].apply(
        lambda g: [x.strip() for x in g.split(",") if x.strip()]
    )

    if "RATING" in df.columns:
        df["RATING"] = pd.to_numeric(df["RATING"], errors="coerce").fillna(0)
    if "VOTES" in df.columns:
        df["VOTES"] = df["VOTES"].astype(str).str.replace(r"[^0-9]", "", regex=True)
        df["VOTES"] = pd.to_numeric(df["VOTES"], errors="coerce").fillna(0)

    return df.reset_index(drop=True)


def build_recommender(df: pd.DataFrame):
    corpus = (
        df["ONE-LINE"].fillna("")
        + " "
        + df["GENRE"].fillna("")
        + " "
        + df.get("STARS", "").fillna("")
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=15000)
    tfidf_matrix = tfidf.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df["MOVIES"].str.lower()).drop_duplicates()
    return cosine_sim, indices


def recommend(title: str, df: pd.DataFrame, cosine_sim, indices, top_n: int = 5):
    title_lower = title.strip().lower()
    if title_lower not in indices:
        matches = [t for t in indices.index if title_lower in t]
        if not matches:
            return None, f"'{title}' not found in the dataset."
        title_lower = matches[0]

    idx = indices[title_lower]
    sim_scores = sorted(
        list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True
    )
    sim_scores = [s for s in sim_scores if s[0] != idx][: top_n * 3]
    movie_indices = [s[0] for s in sim_scores]
    results = df.iloc[movie_indices][["MOVIES", "GENRE", "RATING"]].copy()
    results["ContentSimilarity"] = [round(s[1], 3) for s in sim_scores]
    return results, None


def build_classifier(df: pd.DataFrame):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["GENRE_LIST"])
    tfidf = TfidfVectorizer(stop_words="english", max_features=15000)
    X = tfidf.fit_transform(df["ONE-LINE"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = OneVsRestClassifier(LinearSVC(max_iter=2000))
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    return clf, tfidf, mlb, score


def predict_genre(plot: str, clf, tfidf, mlb):
    X = tfidf.transform([plot])
    y_pred = clf.predict(X)
    genres = mlb.inverse_transform(y_pred)
    return list(genres[0]) if genres[0] else ["Unknown"]


def load_user_ratings(path: str = USER_RATINGS_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["user_id", "MOVIES", "rating", "timestamp"])

    if "MOVIES" in df.columns:
        df["MOVIES"] = df["MOVIES"].astype(str).str.strip()
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    return df


def save_user_rating(movie: str, rating: float, user_id: str = "default", path: str = USER_RATINGS_PATH) -> pd.DataFrame:
    movie = movie.strip()
    ratings_df = load_user_ratings(path)
    timestamp = datetime.now().isoformat()

    same_movie = ratings_df[
        (ratings_df["user_id"] == user_id)
        & (ratings_df["MOVIES"].str.lower() == movie.lower())
    ]
    if not same_movie.empty:
        ratings_df.loc[same_movie.index, ["rating", "timestamp"]] = [rating, timestamp]
    else:
        ratings_df = pd.concat(
            [
                ratings_df,
                pd.DataFrame(
                    [
                        {
                            "user_id": user_id,
                            "MOVIES": movie,
                            "rating": rating,
                            "timestamp": timestamp,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    ratings_df.to_csv(path, index=False)
    return ratings_df


def build_rating_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    if ratings_df.empty:
        return pd.DataFrame()
    return ratings_df.pivot_table(index="user_id", columns="MOVIES", values="rating")


def collaborative_recommend(title: str, df: pd.DataFrame, ratings_df: pd.DataFrame, top_n: int = 5):
    if ratings_df.empty or ratings_df["user_id"].nunique() < 2:
        return None, "Not enough user rating history for collaborative filtering."

    ratings_pivot = build_rating_matrix(ratings_df)
    item_matrix = ratings_pivot.fillna(0).T
    lookup = {movie.lower(): movie for movie in item_matrix.index}
    title_lower = title.strip().lower()
    if title_lower not in lookup:
        similarities = [m for m in lookup if title_lower in m]
        if not similarities:
            return None, f"'{title}' not found in collaborative rating history."
        title_lower = similarities[0]

    movie_name = lookup[title_lower]
    idx = list(item_matrix.index).index(movie_name)
    item_sim = cosine_similarity(item_matrix)
    sim_scores = sorted(
        list(enumerate(item_sim[idx])), key=lambda x: x[1], reverse=True
    )
    sim_scores = [s for s in sim_scores if s[0] != idx][: top_n * 3]
    movie_names = [item_matrix.index[s[0]] for s in sim_scores]

    filtered = df[df["MOVIES"].isin(movie_names)][["MOVIES", "GENRE", "RATING"]].copy()
    filtered = filtered.drop_duplicates(subset=["MOVIES"]).head(top_n * 3)
    filtered["CollaborativeSimilarity"] = [round(s[1], 3) for s in sim_scores[: len(filtered)]]
    return filtered, None


def hybrid_recommend(
    title: str,
    df: pd.DataFrame,
    cosine_sim,
    indices,
    ratings_df: pd.DataFrame,
    top_n: int = 5,
):
    content_results, err = recommend(title, df, cosine_sim, indices, top_n=top_n * 5)
    if err:
        return None, err

    collab_results, collab_err = collaborative_recommend(title, df, ratings_df, top_n=top_n * 5)
    content_map = {row.MOVIES: row.ContentSimilarity for _, row in content_results.iterrows()}
    collab_map = {}
    if collab_results is not None and not collab_err:
        collab_map = {row.MOVIES: row.CollaborativeSimilarity for _, row in collab_results.iterrows()}

    min_rating = df["RATING"].min() if "RATING" in df.columns else 0
    max_rating = df["RATING"].max() if "RATING" in df.columns else 1
    rating_range = max(max_rating - min_rating, 1)

    scored = []
    for _, row in content_results.iterrows():
        movie = row.MOVIES
        content_score = float(row.ContentSimilarity)
        collab_score = float(collab_map.get(movie, 0))
        rating_score = (float(row.RATING) - min_rating) / rating_range
        total = content_score * 0.55 + collab_score * 0.30 + rating_score * 0.15
        scored.append(
            {
                "MOVIES": movie,
                "GENRE": row.GENRE,
                "RATING": row.RATING,
                "ContentSimilarity": round(content_score, 3),
                "CollaborativeSimilarity": round(collab_score, 3),
                "HybridScore": round(total, 3),
            }
        )

    scored_df = pd.DataFrame(scored).drop_duplicates(subset=["MOVIES"]).sort_values(
        by=["HybridScore", "ContentSimilarity", "CollaborativeSimilarity"],
        ascending=False,
    ).head(top_n)
    return scored_df.reset_index(drop=True), None


def search_movies(query: str, df: pd.DataFrame, max_results: int = 20) -> pd.DataFrame:
    if not query:
        return df[["MOVIES", "GENRE", "RATING"]].head(max_results).copy()
    query_lower = query.strip().lower()
    matched = df[
        df["MOVIES"].str.lower().str.contains(query_lower, na=False)
        | df["GENRE"].str.lower().str.contains(query_lower, na=False)
        | df["ONE-LINE"].str.lower().str.contains(query_lower, na=False)
    ]
    return matched[["MOVIES", "GENRE", "RATING"]].drop_duplicates(subset=["MOVIES"]).head(max_results).copy()


def resolve_movie_title(query: str, df: pd.DataFrame, indices=None):
    normalized = query.strip().lower()
    if indices is not None and normalized in indices:
        return df.loc[indices[normalized], "MOVIES"]
    matches = df[df["MOVIES"].str.lower().str.contains(normalized, na=False)]
    if not matches.empty:
        return matches.iloc[0]["MOVIES"]
    return None


class MovieRecommender:
    """Shared recommendation module with support for content, collaborative, and hybrid recommendations."""

    def __init__(
        self,
        data_path: str = DATA_PATH,
        ratings_path: str = USER_RATINGS_PATH,
        train_classifier: bool = True,
    ):
        self.data_path = data_path
        self.ratings_path = ratings_path
        self.df = load_and_clean(self.data_path)
        self.ratings_df = load_user_ratings(self.ratings_path)
        self.cosine_sim, self.indices = build_recommender(self.df)
        self.classifier = None
        self.tfidf_clf = None
        self.mlb = None
        self.classifier_score = None
        if train_classifier:
            self.classifier, self.tfidf_clf, self.mlb, self.classifier_score = build_classifier(
                self.df
            )

    def recommend(self, title: str, top_n: int = 5):
        return recommend(title, self.df, self.cosine_sim, self.indices, top_n)

    def collaborative_recommend(self, title: str, top_n: int = 5):
        return collaborative_recommend(title, self.df, self.ratings_df, top_n)

    def hybrid_recommend(self, title: str, top_n: int = 5):
        return hybrid_recommend(
            title,
            self.df,
            self.cosine_sim,
            self.indices,
            self.ratings_df,
            top_n,
        )

    def predict_genre(self, plot: str):
        if self.classifier is None:
            raise RuntimeError("Genre classifier has not been trained.")
        return predict_genre(plot, self.classifier, self.tfidf_clf, self.mlb)

    def search_movies(self, query: str, max_results: int = 20):
        return search_movies(query, self.df, max_results)

    def save_rating(self, movie: str, rating: float, user_id: str = "default") -> pd.DataFrame:
        self.ratings_df = save_user_rating(movie, rating, user_id, self.ratings_path)
        return self.ratings_df

    def refresh_ratings(self):
        self.ratings_df = load_user_ratings(self.ratings_path)
        return self.ratings_df

    def resolve_movie_title(self, query: str) -> Optional[str]:
        return resolve_movie_title(query, self.df, self.indices)
