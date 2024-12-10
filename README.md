This project is an Interactive Chatbot developed using Natural Language Processing (NLP) techniques. 
The chatbot is capable of understanding user inputs, predicting intents, and providing relevant responses.
It is designed to simulate human-like conversation for use cases such as providing information about recipes, job markets, favorite places, hospitals, and more.
#structure:
│── templates/
│   ├── index.html       
├── app.py                      
├── chatbot.py
├── intents.json                  
├── README.md            
└── requirements.txt  
#Installation
1. Clone the Repository
   git clone <repository-url>
   cd <repository-directory>
2.Set Up a Virtual Environment
   python -m venv venv
   source venv/bin/activate
3.Download NLTK Data
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
#To run the chatbot application locally:
python app.py
#Open the chatbot interface in your web browser at:
  http://127.0.0.1:5000/

   


