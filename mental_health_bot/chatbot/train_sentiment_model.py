import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 📝 Extended Dataset
data = {
    'text': [
        "hello", "hi", "hey", "good morning", "good evening",    # greetings
        "feeling anxious", "i am nervous", "i have anxiety",      # anxiety
        "i feel sad", "i am depressed", "feeling low",            # sadness
        "what is the meaning of life?", "tell me a joke", "who are you?", "what is quantum physics?", "explain universe"  # unknown
    ],
    'emotion': [
        "greeting", "greeting", "greeting", "greeting", "greeting",
        "anxious", "anxious", "anxious",
        "sad", "sad", "sad",
        "unknown", "unknown", "unknown", "unknown", "unknown"
    ]
}

df = pd.DataFrame(data)

# 🔥 Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['emotion']

# 🔥 Model
model = MultinomialNB()
model.fit(X, y)

# 🔥 Save
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ New model trained and saved successfully!")