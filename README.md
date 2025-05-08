🧠 Mental Health Support Chatbot (AIML + ML + Django)
This project is an intelligent Mental Health Support Application developed using Python Django, AIML (Artificial Intelligence Markup Language), and Machine Learning. It simulates a chatbot that offers mental health support by recognizing user emotions like anxiety, sadness, or anger, and responding empathetically with helpful messages and resources.

🌟 Features
🤖 AIML-Based Responses: Uses AIML files for pattern-matching and chatbot replies.

🧠 ML-Based Emotion Detection: A trained sentiment analysis model to detect emotional states.

🔗 Resource Recommendations: Recommends external links based on detected emotion.

💬 Fallback Handling: If AIML fails, ML-based response is used with a friendly fallback message.

🌐 Web Interface: Developed using Django and HTML/CSS frontend.

📦 Modular Architecture: Clean separation of chatbot logic, training scripts, and resource handling.

🚀 How It Works
User sends a message.

The chatbot first attempts to match the input using AIML pattern rules.

If no AIML response is found:

The input is passed to a machine learning model trained on emotional labels.

The model predicts the emotion and the bot replies with a predefined response.

Helpful links are shown based on the emotion category.

Responses are returned asynchronously using AJAX.

🏗️ Project Structure

mental_health_chatbot/
├── chatbot/
│   ├── templates/
│   │   └── chatbot.html         # Chat UI
│   ├── aiml_files/              # AIML knowledge base
│   ├── resources.py             # Helpful links categorized by emotion
│   ├── views.py                 # Core chatbot logic (AIML + ML)
│   ├── sentiment_model.pkl      # Trained sentiment classifier
│   ├── vectorizer.pkl           # TF-IDF Vectorizer
├── train_sentiment_model.py     # Script to train ML model with basic inputs
├── generate_and_train_model.py  # Script to auto-train model from AIML files
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation

📁 Key Files Explained

# views.py
Integrates AIML-based matching and machine learning fallback.

Loads the trained model and TF-IDF vectorizer.

Responds with emotion-specific messages and helpful links.

# train_sentiment_model.py
Trains a basic Naive Bayes classifier using hand-labeled data.

Saves sentiment_model.pkl and vectorizer.pkl.

# generate_and_train_model.py
Auto-parses AIML files to extract patterns and classify them into emotions.

Dynamically trains and saves a model based on AIML content.

🧪 Sample Emotions Detected

Input	Detected Emotion	Response
"I am feeling anxious"	Anxious	"I'm here for you. 🌈 Remember to breathe deeply and stay strong!"
"I feel sad today"	Sad	"It's okay to feel sad sometimes. 💙 I'm here with you."
"Hey there!"	Greeting	"Hello! How can I help you with your mental health today?"
"What is quantum physics?"	Unknown	Fallback response with suggestions to rephrase

🧰 Setup Instructions
# Clone the repository

git clone https://github.com/yourusername/mental-health-chatbot.git
cd mental-health-chatbot

# Create virtual environment and install dependencies

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run training scripts

python train_sentiment_model.py
# or
python generate_and_train_model.py

# Run the Django server
python manage.py runserver

📚 Requirements : 
Django
python-aiml
scikit-learn
joblib
pandas

# You can install them using:
pip install -r requirements.txt

📦 Future Enhancements

🌈 Real-time mood tracking and visualization

🧩 Integration with voice-based input (speech-to-text)

📈 Analytics dashboard for emotion trends

🤝 Connect to mental health professionals via the app

❤️ Contribution
Contributions, issues, and feature requests are welcome!
Feel free to check the issues page.

✨ Acknowledgements
AIML by Dr. Richard Wallace
"Inspired by the need for accessible, empathetic mental health support tools".

💬 Example Chat Flow
User: "I feel very low these days"

Chatbot: "It's okay to feel sad sometimes. 💙 I'm here with you."
(+ Mental health resources linked)

🧪 Model Training Details
Text Classification Model: Naive Bayes

Feature Extraction: TF-IDF / Count Vectorizer

Training Data: Pre-labeled sentences from user emotions and AIML tags

🔐 Privacy & Ethics
This chatbot does not store any personal data and is designed only for supportive and educational use, not a substitute for professional therapy or counseling.

🙌 Contributors
[Mr. Rajarshi Yadav] — Developer & ML Engineer