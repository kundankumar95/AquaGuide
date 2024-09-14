import random
import json
import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize stemmer and download required NLTK data
stemmer = PorterStemmer()
nltk.download("punkt")

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out  # Ensure you return the tensor

def tokenize(sentence):
    """
    Split sentence into array of words/tokens.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stem the word to its root form.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array: 1 for each known word that exists in the sentence, 0 otherwise.
    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and model data
with open('backend/chat/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "backend/chat/data1.pth"
try:
    data = torch.load(FILE, map_location=torch.device('cpu'))
except FileNotFoundError:
    print(f"Error: The file '{FILE}' does not exist.")
    exit()

input_size = data.get("input_size")
hidden_size = data.get("hidden_size")
output_size = data.get("output_size")
all_words = data.get('all_words')
tags = data.get('tags')
model_state = data.get("model_state")

if None in (input_size, hidden_size, output_size, all_words, tags, model_state):
    print("Error: Missing data in the loaded file.")
    exit()

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def Chat(st: str):
    """
    Chat function to process user input and provide a response.
    """
    # Tokenize and process the input
    sentence = tokenize(st)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Forward pass through the model
    with torch.no_grad():  # Disable gradient calculation
        output = model(X)
    
    # Debug prints
    print("Model output:", output)
    
    # Ensure output is valid
    if output is None or output.shape[0] == 0:
        return "I do not understand...", None

    # Get the predicted class
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Get the probability of the prediction
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Find the response for the predicted tag
    for intent in intents['intents']:
        if tag == intent["tag"]:
            return random.choice(intent['responses']), prob.item()
    
    return "I do not understand...", None

if __name__ == "__main__":
    query = input("Enter Your query: ")
    response, prob = Chat(query)
    print(f"Response: {response}")
    print(f"Probability: {prob}")
