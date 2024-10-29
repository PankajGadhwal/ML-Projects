import streamlit as st
import pickle
import torch
import torch.nn as nn
import nltk
import json

nltk.download('punkt')
nltk.download('punkt_tab')

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
    
model_path = "A3/rnn_text_generator.pkl"
vocab_path = "A3/vocab.json"

with open(model_path, "rb") as f:
    model = pickle.load(f)
model.eval()  

with open(vocab_path, "r") as f:
    vocab_data = json.load(f)
word_to_idx = vocab_data["word_to_idx"]
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

def preprocess_text(sentence):
    tokens = nltk.tokenize.word_tokenize(sentence.lower())
    words = [word for word in tokens if word.isalpha()]
    return words

def predict_next_k_words(model, sentence, word_to_idx, idx_to_word, context_length, k=1):
    words = preprocess_text(sentence)
    predicted_words = []

    if len(words) < context_length:
        if 'padding' in word_to_idx:
            words = ['padding'] * (context_length - len(words)) + words
        else:
            words = ['unknown token'] * (context_length - len(words)) + words

    for _ in range(k):
        input_indices = [word_to_idx.get(word, word_to_idx['unknown token']) for word in words[-context_length:]]
        input_tensor = torch.tensor([input_indices], dtype=torch.long)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = torch.argmax(output, dim=-1).item()

        predicted_word = idx_to_word.get(predicted_idx, 'unknown token')
        if predicted_word == 'unknown token':
            print("Predicted word is 'unknown token' or empty, stopping prediction.")
            break

        predicted_words.append(predicted_word)
        words.append(predicted_word)

    return predicted_words

st.title("Next-Word Prediction App")
st.write("Enter a sentence and select the number of words you'd like the model to predict.")
sentence = st.text_input("Enter the starting sentence:", value="I thought today's fete had been")
k = st.slider("Select the number of words to predict (k):", min_value=1, max_value=10, value=2)

if st.button("Predict Next Words"):
    try:
        predicted_words = predict_next_k_words(model, sentence, word_to_idx, idx_to_word, context_length=5, k=k)
        st.write(f"The next {k} words are: {' '.join(predicted_words)}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
