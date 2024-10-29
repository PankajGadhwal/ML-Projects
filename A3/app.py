'''import streamlit as st
import pickle
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize

# Load the trained model
with open('mlp_text_generator.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the vocabulary
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

class MLPTextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MLPTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.activation = nn.ReLU()  # Change this to another activation function if needed

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Average over the sequence length
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
# Create index-to-word mapping
idx_to_word = {i: word for word, i in vocab.items()}

# Define the prediction function
def predict_next_words(model, user_input, vocab, idx_to_word, context_length, num_predictions):
    # Tokenize user input
    user_tokens = word_tokenize(user_input.lower())
    
    if len(user_tokens) >= context_length:
        input_sequence = [vocab[word] for word in user_tokens[-context_length:]]
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0)  # Reshape for batch
        with torch.no_grad():
            model.eval()
            output = model(input_tensor)
            _, predicted_idx = torch.topk(output, num_predictions, dim=1)
            predicted_words = [idx_to_word[idx] for idx in predicted_idx[0].tolist()]
        return predicted_words
    else:
        return ["Not enough input for prediction."]

# Streamlit app title
st.title("Next-Word Prediction App")

# Sidebar for parameters
context_length = st.sidebar.slider("Context Length", 2, 10, value=5)
num_predictions = st.sidebar.slider("Number of Words to Predict", 1, 10, value=5)

# User input
user_input = st.text_input("Enter a sentence:", "This is an example.")

# Prediction logic
if st.button("Predict"):
    predicted_words = predict_next_words(model, user_input, vocab, idx_to_word, context_length, num_predictions)
    st.write(f"Predicted next words: {', '.join(predicted_words)}")'''

# streamlit_app.py

import streamlit as st
import pickle
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import json

class RNNTextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x) 
        x = self.fc(x[:, -1, :])
        return x
    
# Load the saved model and vocabulary
MODEL_PATH = "rnn_text_generator.pkl"
VOCAB_PATH = "vocab.json"

# Load the model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
model.eval()  # Set model to evaluation mode

# Load vocabulary mappings
with open(VOCAB_PATH, "r") as f:
    vocab_data = json.load(f)
    print(vocab_data)  # Optional for debugging
word_to_idx = vocab_data["word_to_idx"]
idx_to_word = vocab_data["idx_to_word"]
# Function to predict the next k words
def predict_next_k_words(model, sentence, word_to_idx, idx_to_word, context_length, k=1):
    words = sentence.lower().split()
    predicted_words = []
    if len(words) < context_length:
        words = ['padding'] * (context_length - len(words)) + words
    
    for _ in range(k):
        # Prepare input for model
        input_indices = [word_to_idx.get(word, word_to_idx['unknown token']) for word in words[-context_length:]]
        input_tensor = torch.tensor([input_indices], dtype=torch.long)

        # Get model output and predicted word
        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = torch.argmax(output, dim=-1).item()
            predicted_word = idx_to_word[predicted_idx]

        # Add the predicted word to the sentence
        predicted_words.append(predicted_word)
        words.append(predicted_word)
        
    return predicted_words

# Streamlit UI setup
st.title("Next-Word Prediction App")
st.write("Enter a sentence and select the number of words you'd like the model to predict.")

# User inputs
sentence = st.text_input("Enter the starting sentence:", value="I want to say you")
k = st.slider("Select the number of words to predict (k):", min_value=1, max_value=10, value=3)

# Button to generate prediction
if st.button("Predict Next Words"):
    # Predict and display the result
    predicted_words = predict_next_k_words(model, sentence, word_to_idx, idx_to_word, context_length=5, k=k)
    st.write(f"The next {k} words are: {' '.join(predicted_words)}")
