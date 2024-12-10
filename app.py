from flask import Flask, render_template, request, jsonify
import nltk
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load intents.json file
with open(r'D:\nlp\intents.json', 'r') as file:
    intents = json.load(file)

data = []
labels = []

for intent in intents:
    for pattern in intent["patterns"]:
        data.append(pattern)
        labels.append(intent["tag"])

# Train-test split
X = np.array(data)
y = np.array(labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a machine learning model pipeline
model = make_pipeline(CountVectorizer(), LogisticRegression())

# Train model
model.fit(X_train, y_train)

# Function to return a chatbot response based on user intent
def get_response(intent_tag):
    for intent in intents:
        if intent["tag"] == intent_tag:
            return np.random.choice(intent["responses"])
    return "I don't understand that."

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle chatbot communication
@app.route("/get_response", methods=["POST"])
def chatbot_response():
    user_input = request.form["user_input"]
    predicted_tag = model.predict([user_input])[0]
    response = get_response(predicted_tag)
    return jsonify({"response": response})


# Start the Flask server
if __name__ == "__main__":
    app.run(debug=True)

