

import pandas as pd
import nltk
import warnings
from fuzzywuzzy import process
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load Dataset
df = pd.read_csv("/content/drive/MyDrive/books.csv", sep=";", encoding="ISO-8859-1", on_bad_lines="skip")

# Remove books with missing or very short titles/authors
df = df[['Book-Title', 'Book-Author', 'Year-Of-Publication']].dropna()
df = df[df['Book-Title'].str.len() > 3]  # Remove invalid short book titles

# **Manually Define Genre Mappings**
genre_keywords = {
    "fantasy": ["fantasy", "magic", "dragon", "wizard", "sorcerer", "myth", "fairy", "spell"],
    "mystery": ["mystery", "detective", "crime", "thriller", "suspense", "investigation", "sherlock", "murder"],
    "romance": ["romance", "love", "passion", "relationship", "affair"],
    "adventure": ["adventure", "journey", "explore", "action", "quest", "expedition", "voyage", "treasure"],
    "horror": ["horror", "scary", "ghost", "haunted", "fear", "dark", "nightmare"],
    "science fiction": ["sci-fi", "space", "aliens", "technology", "future", "robots", "galaxy"],
    "history": ["history", "biography", "war", "ancient", "historical"],
}

# **Assign genres to books**
df['Genre'] = "unknown"
for genre, keywords in genre_keywords.items():
    df.loc[df['Book-Title'].str.contains('|'.join(keywords), case=False, na=False), 'Genre'] = genre

# **TF-IDF Vectorizer for Similarity Matching**
vectorizer = TfidfVectorizer(stop_words='english')
book_vectors = vectorizer.fit_transform(df['Book-Title'] + " " + df['Book-Author'])

# **Function to preprocess text**
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return ' '.join(filtered_tokens)

# **Function to find the closest book title**
def get_closest_book_title(user_title):
    matches = process.extractOne(user_title, df["Book-Title"].values, score_cutoff=85)
    return matches[0] if matches else None

# **Function to find the closest author name**
def get_closest_author(user_author):
    matches = process.extractOne(user_author, df["Book-Author"].values, score_cutoff=85)
    return matches[0] if matches else None

# **Function to recommend books based on title, genre, or author**
def recommend_books(user_input, num_recommendations=5):
    processed_input = preprocess_text(user_input)

    # **Step 1: Check if input is a genre**
    matching_genres = [genre for genre, keywords in genre_keywords.items() if any(word in processed_input for word in keywords)]

    if matching_genres:
        genre = matching_genres[0]
        filtered_books = df[df['Genre'] == genre]
        if not filtered_books.empty:
            return filtered_books['Book-Title'].sample(n=min(num_recommendations, len(filtered_books))).tolist()
        else:
            return ["No books found for that genre. Try another genre!"]

    # **Step 2: Check if input is an author**
    matched_author = get_closest_author(processed_input)
    if matched_author:
        author_books = df[df['Book-Author'] == matched_author]
        return author_books['Book-Title'].sample(n=min(num_recommendations, len(author_books))).tolist()

    # **Step 3: If no author match, try matching book title**
    matched_title = get_closest_book_title(processed_input)
    if not matched_title:
        return ["I couldn't find that book. Try another title."]

    # **Step 4: Find similar books (ONLY within the same genre)**
    matched_genre = df[df['Book-Title'] == matched_title]['Genre'].values[0]
    genre_books = df[df['Genre'] == matched_genre]

    if not genre_books.empty:
        genre_vectors = vectorizer.transform(genre_books['Book-Title'] + " " + genre_books['Book-Author'])
        idx = genre_books.index[genre_books["Book-Title"] == matched_title].tolist()[0]
        scores = list(enumerate(cosine_similarity(genre_vectors[idx], genre_vectors)[0]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

        book_indices = [i[0] for i in scores]
        return [matched_title] + genre_books["Book-Title"].iloc[book_indices].tolist()

    return ["No similar books found. Try another title!"]

# **Chatbot Function**
def chatbot():
    print("\n📚 Welcome to BookBot! Tell me your favorite book, genre, or author, and I'll suggest some books!")

    while True:
        user_input = input("\nYou: ").strip().lower()

        if user_input in ["exit", "quit", "bye"]:
            print("BookBot: Goodbye! Happy reading! 📚")
            break

        if user_input in ["hello", "hi", "hey"]:
            print("BookBot: Hello! How can I help you?")
            continue

        recommendations = recommend_books(user_input)
        print("\n📖 BookBot: Here are some book recommendations:")
        for book in recommendations:
            print(f"- {book}")

if __name__ == "__main__":
    chatbot()
