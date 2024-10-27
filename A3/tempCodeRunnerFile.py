import streamlit as st
import pickle
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the vocabulary
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

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
    st.write(f"Predicted next words: {', '.join(predicted_words)}")