# 🎌 AI-Powered Anime Recommender System

An intelligent anime recommendation system that uses advanced natural language processing and semantic search to suggest personalized anime based on user descriptions. Built with Flask, sentence transformers, and FAISS for efficient similarity search.

## ✨ Features

- **🧠 AI-Powered Recommendations**: Uses sentence transformers to understand user preferences and find semantically similar anime
- **⚡ Real-time Search**: Instant recommendations based on natural language descriptions
- **🖼️ Visual Interface**: Clean, responsive web interface with anime artwork
- **📊 Smart Matching**: Fuzzy string matching for anime titles and images
- **🎯 Accurate Results**: FAISS indexing for fast and precise similarity search
- **📱 Mobile Responsive**: Works seamlessly across all devices

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **ML/AI**: Sentence Transformers, FAISS
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Text Processing**: FuzzyWuzzy for string matching
- **Data Processing**: Pandas, NumPy
- **Alternative Interface**: Streamlit (included)

## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/anime-recommender-system.git
   cd anime-recommender-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place your anime dataset (CSV format) in the project directory
   - Update the dataset path in `llm_vectorSearch.py` if needed
   - Add anime images to `static/images/` folder

4. **Initialize embeddings** (First time only)
   ```python
   # Uncomment the relevant lines in llm_vectorSearch.py to process your dataset
   python llm_vectorSearch.py
   ```

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
flask==2.3.3
sentence-transformers==2.2.2
faiss-cpu==1.7.4
fuzzywuzzy==0.18.0
python-levenshtein==0.21.1
pandas==2.0.3
numpy==1.24.3
streamlit==1.28.1
scikit-learn==1.3.0
```

## 🎮 Usage

### Web Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Describe your preferences** in natural language, such as:
   - "I want a romantic comedy anime with school setting"
   - "Looking for action anime with supernatural powers"
   - "Suggest slice of life anime with friendship themes"

### Streamlit Alternative

For a simpler interface, you can also run the Streamlit version:

```bash
streamlit run llm_vectorSearch.py
```

## 📁 Project Structure

```
anime-recommender-system/
│
├── app.py                      # Main Flask application
├── llm_vectorSearch.py         # Core ML system & Streamlit app
├── text_embedding_system.py    # Embedding system class
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Home page
│   └── user.html              # Search/results page
│
├── static/                     # Static assets
│   ├── images/                # Anime artwork
│   └── image1.png             # Hero image
│
└── embeddings/                 # Generated embeddings (auto-created)
    ├── faiss_index.index      # FAISS search index
    └── embeddings.pkl         # Processed embeddings
```

## 🔧 Configuration

### Dataset Format

Your anime dataset should be a CSV file with a `combined_info` column containing:
```
Title: [Anime Title]. Overview: [Synopsis and genre information]
```

### Image Matching

- Place anime images in `static/images/`
- Name images to match anime titles for automatic association
- System uses fuzzy matching with 60% similarity threshold
- Falls back to `default_anime.jpg` if no match found

## 📸 Screenshots

### Home Page
![image](https://github.com/user-attachments/assets/2f4f50b9-36ce-453b-99dd-f579b4e9bbd2)
![image](https://github.com/user-attachments/assets/579ec650-c331-47b5-95f6-5dfa47e90ecb)
![image](https://github.com/user-attachments/assets/a4e6ef0f-c2c1-4259-82b9-4a9e7b2c7778)

### Search Interface
![image](https://github.com/user-attachments/assets/d1b1c6fb-fe59-4dca-a16d-fdeffebddee8)

### Recommendation Results
![image](https://github.com/user-attachments/assets/e3b30775-99fa-453c-a985-0b79a5bb0b96)
![image](https://github.com/user-attachments/assets/a9818095-950c-499d-9882-9f93788f5cea)


## 🤖 How It Works

1. **Text Processing**: User descriptions are processed using sentence transformers to create semantic embeddings
2. **Similarity Search**: FAISS performs efficient similarity search against pre-computed anime embeddings
3. **Ranking**: Results are ranked by semantic similarity scores
4. **Image Matching**: Fuzzy string matching associates anime titles with corresponding artwork
5. **Display**: Results are presented with titles, ratings, genres, synopses, and images

## 🎯 Model Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Search Index**: FAISS L2 distance for similarity matching
- **Text Matching**: FuzzyWuzzy ratio-based string similarity
- **Preprocessing**: Custom text parsing for title and overview extraction

## 🔄 Data Pipeline

1. **Data Ingestion**: Load anime dataset from CSV
2. **Text Preprocessing**: Extract titles and overviews
3. **Embedding Generation**: Create semantic embeddings using sentence transformers
4. **Index Building**: Build FAISS index for efficient search
5. **Persistence**: Save embeddings and index for future use

## 🚀 Deployment

For production deployment:

1. **Disable GPU** (already configured):
   ```python
   os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
   ```

2. **Set production configuration**:
   ```python
   app.run(host='0.0.0.0', port=5000, debug=False)
   ```

3. **Use production WSGI server** like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sentence Transformers** for powerful semantic embeddings
- **FAISS** for efficient similarity search
- **Flask** for the web framework
- **Bootstrap** for responsive design components
- **Font Awesome** for icons

## 📞 Support

If you encounter any issues or have questions:

1. Check existing [Issues](https://github.com/yourusername/anime-recommender-system/issues)
2. Create a new issue with detailed description
3. Include error messages and system information

## 🔮 Future Enhancements

- [ ] User authentication and preference storage
- [ ] Advanced filtering options (genre, year, rating)
- [ ] Integration with anime databases (MAL, AniList)
- [ ] Collaborative filtering recommendations
- [ ] Multi-language support
- [ ] API endpoints for external integration

---

⭐ **Star this repository if you found it helpful!**
