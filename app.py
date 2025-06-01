from flask import Flask, render_template, request, redirect, url_for, flash
from text_embedding_system import TextEmbeddingSystem
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
from datetime import timedelta

# Ensure GPU is disabled if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "Hello"
app.permanent_session_lifetime = timedelta(minutes=5)

# Initialize the TextEmbeddingSystem
embedding_system = TextEmbeddingSystem()
embedding_system.load_embeddings_and_index()  # Load embeddings and FAISS index

# Path to the folder with anime images
IMAGES_FOLDER = os.path.join('static', 'images')

# Ensure the folder exists and retrieve image filenames
if os.path.exists(IMAGES_FOLDER):
    image_files = os.listdir(IMAGES_FOLDER)
else:
    image_files = []

def find_matching_image(title):
    """Find the best matching image for a given anime title with 60% or higher similarity."""
    if image_files:
        best_match = process.extractOne(title, image_files, scorer=fuzz.ratio)
        if best_match and best_match[1] >= 60:  # 60% similarity threshold
            return best_match[0]  # Return the filename of the best match
    return "default_anime.jpg"  # Default image if no match is found

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/user", methods=["GET", "POST"])
def user():
    recommendations = []
    if request.method == "POST":
        description = request.form.get("description", "").strip()

        if description:
            # Convert the user's description into embeddings and search
            results = embedding_system.search(description, k=8)
            recommendations = []
            for result in results:
                title = result[0]
                # The synopsis and genre are in result[2]
                synopsis_genres = result[2]
                synopsis = ""
                genres = "N/A"

                # Attempt to split synopsis and genres
                if "Genres: " in synopsis_genres:
                    parts = synopsis_genres.split("Genres: ", 1)
                    synopsis = parts[0].strip()
                    genres = parts[1].strip()
                else:
                     synopsis = synopsis_genres.strip()

                recommendations.append({
                    "title": title,
                    "rating": f"{10 - result[1]:.2f}",
                    "Synopsis": synopsis,  # Store only synopsis here
                    "genres": genres,      # Store genres separately
                    "image": find_matching_image(title)
                })
        
        if not recommendations:
            flash("No anime found matching your description. Showing random recommendations.", "info")
            recommendations = [
                {
                    "title": "Random Anime",
                    "rating": "N/A",
                    "Synopsis": "Random Synopsis",
                    "genres": "Random Genre",
                    "image": "default_anime.jpg",
                }
                for _ in range(8)
            ]

    return render_template("user.html", recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
