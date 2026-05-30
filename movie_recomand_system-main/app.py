import argparse
import os
from flask import Flask, request, render_template_string, flash
from movie_recommender import MovieRecommender

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Movie Recommendation System</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f4f7fb; }
    .container { max-width: 980px; margin: 24px auto; padding: 24px; background: white; border-radius: 8px; box-shadow: 0 0 20px rgba(0,0,0,0.08); }
    h1 { margin-top: 0; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    label { display:block; font-weight: 600; margin-bottom: 6px; }
    input[type=text], input[type=number], textarea, select { width:100%; padding:8px; margin-bottom:12px; border:1px solid #cbd5e1; border-radius:4px; }
    button { background:#2f6fdb; color:white; border:none; padding:10px 18px; border-radius:4px; cursor:pointer; }
    button:hover { background:#214f9f; }
    .results { margin-top:24px; }
    table { width:100%; border-collapse:collapse; }
    th, td { text-align:left; padding:10px; border-bottom:1px solid #e2e8f0; }
    th { background:#f8fafc; }
    .message { margin-bottom:16px; padding:12px; border-radius:4px; }
    .error { background:#ffe3e3; color:#861b1b; }
    .success { background:#ddf7e5; color:#14532d; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Movie Recommendation System</h1>
    <p>Use the forms below to search movies, get recommendations, rate titles, or predict genres.</p>
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for category, message in messages %}
          <div class="message {{ category }}">{{ message }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}
    <div class="grid">
      <form method="post">
        <h2>Search</h2>
        <label for="query">Movie title, genre, or plot keyword</label>
        <input id="query" name="query" type="text" value="{{ query | default('') }}" placeholder="e.g. sci-fi, crime, space" />
        <button name="action" value="search">Search Movies</button>
      </form>

      <form method="post">
        <h2>Get Recommendations</h2>
        <label for="title">Movie title</label>
        <input id="title" name="title" type="text" value="{{ title | default('') }}" placeholder="e.g. Inception" />
        <label for="rec_type">Recommendation type</label>
        <select id="rec_type" name="rec_type">
          <option value="content" {% if rec_type=='content' %}selected{% endif %}>Content-based</option>
          <option value="collaborative" {% if rec_type=='collaborative' %}selected{% endif %}>Collaborative</option>
          <option value="hybrid" {% if rec_type=='hybrid' %}selected{% endif %}>Hybrid</option>
        </select>
        <button name="action" value="recommend">Generate Recommendations</button>
      </form>
    </div>

    <div class="grid">
      <form method="post">
        <h2>Rate a Movie</h2>
        <label for="rating_title">Movie title</label>
        <input id="rating_title" name="rating_title" type="text" value="{{ rating_title | default('') }}" placeholder="e.g. The Matrix" />
        <label for="user_id">User ID</label>
        <input id="user_id" name="user_id" type="text" value="{{ user_id | default('default') }}" />
        <label for="rating">Rating (0-10)</label>
        <input id="rating" name="rating" type="number" min="0" max="10" step="0.5" value="{{ rating | default('') }}" />
        <button name="action" value="rate">Save Rating</button>
      </form>

      <form method="post">
        <h2>Genre Prediction</h2>
        <label for="plot">Plot description</label>
        <textarea id="plot" name="plot" rows="6" placeholder="Enter a short plot description">{{ plot | default('') }}</textarea>
        <button name="action" value="genre">Predict Genre</button>
      </form>
    </div>

    {% if result_html %}
      <div class="results">
        <h2>Results</h2>
        {{ result_html | safe }}
      </div>
    {% endif %}

    {% if genre_prediction %}
      <div class="results">
        <h2>Predicted Genres</h2>
        <p>{{ genre_prediction }}</p>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""


def create_app():
    recommender = MovieRecommender()
    app = Flask(__name__)
    app.secret_key = "change-me-for-production"

    def build_table(data_frame):
        if data_frame is None or data_frame.empty:
            return "<p>No results found.</p>"
        return data_frame.to_html(classes="table", index=False, border=0, escape=False)

    @app.route('/', methods=['GET', 'POST'])
    def index():
        query = ''
        title = ''
        rec_type = 'content'
        rating_title = ''
        user_id = 'default'
        rating = ''
        plot = ''
        result_html = ''
        genre_prediction = ''

        if request.method == 'POST':
            action = request.form.get('action', '')
            query = request.form.get('query', '').strip()
            title = request.form.get('title', '').strip()
            rec_type = request.form.get('rec_type', 'content')
            rating_title = request.form.get('rating_title', '').strip()
            user_id = request.form.get('user_id', 'default').strip() or 'default'
            rating = request.form.get('rating', '').strip()
            plot = request.form.get('plot', '').strip()

            if action == 'search':
                result_html = build_table(recommender.search_movies(query))
            elif action == 'recommend':
                if not title:
                    flash('Please enter a movie title for recommendations.', 'error')
                else:
                    if rec_type == 'content':
                        df, err = recommender.recommend(title, top_n=10)
                    elif rec_type == 'collaborative':
                        df, err = recommender.collaborative_recommend(title, top_n=10)
                    else:
                        df, err = recommender.hybrid_recommend(title, top_n=10)
                    if err:
                        flash(err, 'error')
                    else:
                        result_html = build_table(df)
            elif action == 'rate':
                if not rating_title or not rating:
                    flash('Please enter both a movie title and a numeric rating.', 'error')
                else:
                    try:
                        rating_value = float(rating)
                        recommender.save_rating(rating_title, rating_value, user_id)
                        flash(f'Rating saved for {rating_title}.', 'success')
                    except ValueError:
                        flash('Rating must be a number between 0 and 10.', 'error')
            elif action == 'genre':
                if not plot:
                    flash('Please enter a plot description to predict genres.', 'error')
                else:
                    genres = recommender.predict_genre(plot)
                    genre_prediction = ', '.join(genres)

        return render_template_string(
            TEMPLATE,
            query=query,
            title=title,
            rec_type=rec_type,
            rating_title=rating_title,
            user_id=user_id,
            rating=rating,
            plot=plot,
            result_html=result_html,
            genre_prediction=genre_prediction,
        )

    return app


def run_cli():
    recommender = MovieRecommender()
    print('=' * 70)
    print('Movie Recommendation System - CLI Interface')
    print('=' * 70)
    print(f'Loaded {len(recommender.df)} movies from dataset.')
    if recommender.classifier is not None:
        print(f'Genre classifier accuracy: {recommender.classifier_score:.2%}')
    print()

    while True:
        print('Choose an option:')
        print('  1. Search movies')
        print('  2. Content-based recommendations')
        print('  3. Collaborative recommendations')
        print('  4. Hybrid recommendations')
        print('  5. Predict genre from plot')
        print('  6. Save a rating')
        print('  7. Exit')
        choice = input('Enter choice (1-7): ').strip()

        if choice == '1':
            query = input('Search query: ').strip()
            results = recommender.search_movies(query)
            print(results.to_string(index=False))
        elif choice == '2':
            title = input('Movie title: ').strip()
            results, err = recommender.recommend(title, top_n=10)
            if err:
                print(err)
            else:
                print(results.to_string(index=False))
        elif choice == '3':
            title = input('Movie title: ').strip()
            results, err = recommender.collaborative_recommend(title, top_n=10)
            if err:
                print(err)
            else:
                print(results.to_string(index=False))
        elif choice == '4':
            title = input('Movie title: ').strip()
            results, err = recommender.hybrid_recommend(title, top_n=10)
            if err:
                print(err)
            else:
                print(results.to_string(index=False))
        elif choice == '5':
            plot = input('Plot description: ').strip()
            genres = recommender.predict_genre(plot)
            print('Predicted genres:', ', '.join(genres))
        elif choice == '6':
            title = input('Movie title: ').strip()
            user_id = input('User ID [default]: ').strip() or 'default'
            rating = input('Rating (0-10): ').strip()
            try:
                rating_value = float(rating)
                recommender.save_rating(title, rating_value, user_id)
                print(f'Rating saved for {title}.')
            except ValueError:
                print('Invalid rating. Please enter a number between 0 and 10.')
        elif choice == '7':
            print('Goodbye!')
            break
        else:
            print('Invalid selection. Choose a number between 1 and 7.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--web', action='store_true', help='Start the web interface')
    parser.add_argument('--host', default='127.0.0.1', help='Web server host')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    args = parser.parse_args()

    if args.web:
        app = create_app()
        app.run(host=args.host, port=args.port)
    else:
        run_cli()
