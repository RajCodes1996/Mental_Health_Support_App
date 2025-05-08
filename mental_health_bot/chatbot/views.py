import os
import aiml
from django.shortcuts import render
from django.http import JsonResponse
import joblib
import random
from chatbot.resources import resources

# 🧠 Load ML model and vectorizer
BASE_DIR = os.path.dirname(__file__)
sentiment_model = joblib.load(os.path.join(BASE_DIR, "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

# 🧠 Load AIML files
kernel = aiml.Kernel()
AIML_DIR = os.path.join(BASE_DIR, 'aiml_files')

for file in os.listdir(AIML_DIR):
    if file.endswith('.aiml'):
        print("Loading:", file)
        kernel.learn(os.path.join(AIML_DIR, file))

# 🌟 Fallback responses
fallback_responses = [
    "I'm still learning! 🌟 Can you try rephrasing your question?",
    "Sorry, I don't have an answer yet. 🌸 Feel free to ask something else!",
    "I'm working on becoming smarter. 🧠 Can you ask differently?",
    "Hmm... I'm not sure about that yet! 🚀"
]

def chatbot(request):
    if request.method == "POST":
        user_input = request.POST.get("message")

        if user_input:
            # 🎯 First try AIML
            aiml_response = kernel.respond(user_input.upper())

            if aiml_response.strip() != "":
                response = aiml_response
            else:
                # 🔎 Try ML-based prediction
                X_input = vectorizer.transform([user_input])
                predicted_emotion = sentiment_model.predict(X_input)[0]
                predicted_emotion = str(predicted_emotion).lower()

                # 💬 Generate response
                if predicted_emotion == "greeting":
                    response = "Hello! How can I help you with your mental health today?"
                elif predicted_emotion == "anxious":
                    response = "I'm here for you. 🌈 Remember to breathe deeply and stay strong!"
                elif predicted_emotion == "sad":
                    response = "It's okay to feel sad sometimes. 💙 I'm here with you."
                elif predicted_emotion == "angry":
                    response = "Anger is natural, but you're not alone. I'm listening. 💬"
                else:
                    response = random.choice(fallback_responses)

                # 🧩 Add Recommendations (Safely)
                recommended = resources.get(predicted_emotion, resources.get("general", []))
                if recommended:
                    links_text = "\n\nHere are some helpful resources:\n"
                    for item in recommended:
                        links_text += f"🔹 <a href='{item['url']}' target='_blank'>{item['title']}</a><br>"
                    response += links_text

            return JsonResponse({"response": response})
        else:
            return JsonResponse({"response": "Please type a message."})

    return render(request, "chatbot.html")