# FUTURE_ML_03
# AI Chatbot for Book Recommendation

## Task Overview
This project involves building an AI-powered chatbot that recommends books based on user preferences. The chatbot utilizes Natural Language Processing (NLP) techniques to understand user input and provide personalized book suggestions.

## Skills Gained
- Natural Language Processing (NLP)
- Conversational AI
- Recommendation Systems

## Tools and Technologies Used
- **Programming Language:** Python
- **Libraries & Frameworks:**
  - NLTK (Natural Language Toolkit)
  - ChatterBot (Conversational AI)
  - Scikit-learn (Machine Learning)

## Dataset
- **Books Dataset:** [Kaggle Books Dataset](https://www.kaggle.com/datasets/saurabhbagchi/books-dataset)
- The dataset includes book titles, authors, genres, ratings, and other relevant information for recommendation.

## Project Structure
```
📂 AI-Book-Recommendation-Chatbot
│── 📂 data
│   ├── books.csv  # Processed dataset
│── 📂 models
│   ├── chatbot_model.pkl  # Trained chatbot model
│── 📂 scripts
│   ├── preprocess.py  # Data preprocessing script
│   ├── train_chatbot.py  # Model training script
│   ├── chatbot.py  # Chatbot implementation
│── README.md  # Project documentation
```

## How to Run the Chatbot
1. **Install Dependencies:**
   ```sh
   pip install nltk chatterbot chatterbot_corpus scikit-learn pandas
   ```
2. **Prepare the Dataset:**
   ```sh
   python scripts/preprocess.py
   ```
3. **Train the Chatbot:**
   ```sh
   python scripts/train_chatbot.py
   ```
4. **Run the Chatbot:**
   ```sh
   python scripts/chatbot.py
   ```

## Expected Output
- The chatbot interacts with users and suggests books based on their preferences, such as genre, author, or previous reading history.
- Example Interaction:
  ```
  You: Recommend a fantasy book
 BookBot: Here are some book recommendations:
- Sword and Sorceress XII: An Anthology of Heroic Fantasy
- Down and Out in the Magic Kingdom
- Magic School Bus/Boxed Set
- Moreta: Dragonlady of Pern
- Twister On Tuesday (Magic Tree House 23, paper)
  ```

## Deliverables
- A functional book recommendation chatbot.
- A presentation explaining the recommendation logic and chatbot flow.

## Future Enhancements
- Integrate advanced NLP techniques for better intent recognition.
- Expand the dataset for improved recommendations.
- Deploy the chatbot as a web application or API.

## Acknowledgments
- Future Interns for providing the project opportunity.
- Open-source datasets and libraries used in the project.

