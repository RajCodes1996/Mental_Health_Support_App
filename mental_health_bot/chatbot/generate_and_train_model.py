# generate_and_train_model.py
import os
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# 📂 AIML directory
AIML_DIR = os.path.join(os.path.dirname(__file__), 'aiml_files')

# 📄 Initialize data lists
texts = []
labels = []

# 🛠 Function to clean text
def clean_text(text):
    return text.replace("*", "").replace("_", "").strip()

# 📚 Parse AIML files
for file in os.listdir(AIML_DIR):
    if file.endswith('.aiml'):
        tree = ET.parse(os.path.join(AIML_DIR, file))
        root = tree.getroot()

        for category in root.findall('category'):
            pattern = category.find('pattern').text
            template = category.find('template').text

            if pattern and template:
                pattern = clean_text(pattern)
                template = clean_text(template)

                texts.append(pattern)
                
                # 🎯 Simple logic: Tagging based on keywords inside the template
                if "anxious" in template.lower() or "anxiety" in template.lower():
                    labels.append("anxious")
                elif "sad" in template.lower() or "breathe" in template.lower():
                    labels.append("sad")
                elif "hello" in template.lower() or "hi" in template.lower():
                    labels.append("greeting")
                else:
                    labels.append("others")

# 📊 Check if data is generated
if not texts:
    print("⚠️ No data found in AIML files!")
    exit()

# 🧠 Create machine learning pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# 🏋️‍♂️ Train the model
model.fit(texts, labels)

# 💾 Save the model
joblib.dump(model, "sentiment_model.pkl")

print(f"✅ Model trained on {len(texts)} samples and saved successfully!")