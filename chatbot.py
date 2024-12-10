import json
import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
# Load intents from the JSON file
file_path = r'intents.json'  
with open(file_path, 'r') as file:
    intents = json.load(file)
# Prepare the data
data = []
labels = []
for intent in intents:
    for pattern in intent["patterns"]:
        data.append(pattern)
        labels.append(intent["tag"])
# Convert the data into a numpy array for the model
X = np.array(data)
y = np.array(labels)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a pipeline with CountVectorizer and Logistic Regression
model = make_pipeline(CountVectorizer(), LogisticRegression())
# Train the model
model.fit(X_train, y_train)
# Function to generate responses
def generate_response(intent_tag):
    for intent in intents:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
# Function to chat with the bot
def chat():
    print("Bot: Hello! How can I assist you today? (Type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Bot: Goodbye!")
            break
        predicted_tag = model.predict([user_input])[0]
        response = generate_response(predicted_tag)
        print(f"Bot: {response}")
chat()
