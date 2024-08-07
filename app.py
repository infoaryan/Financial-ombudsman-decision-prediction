import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import spacy
import string
import nltk
import streamlit.components.v1 as components
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

nltk.download('stopwords')
nltk.download('punkt')
# nltk.download('omw-1.4')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def spacy_lemmatize(text):
    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc)

def clean_spacy_text(text):
    # Convert to lowercase
    text = str(text)
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = spacy_lemmatize(text)
    # Tokenize text
    words = word_tokenize(text)
    # Remove stop words and apply lemmatization
    words = [word for word in words if word not in stop_words]
    # Join words back into a single string
    text = ' '.join(words)
    return text

# Load the trained model
model = tf.keras.models.load_model('models/model.h5', custom_objects={'Attention': Attention})

# Load tokenizer (you should use the same tokenizer used during training)
import pickle
with open('models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_text_length = 300  # Adjust this to your model's max sequence length

# def preprocess_text(text):
#     sequences = tokenizer.texts_to_sequences([text])
#     padded_sequences = pad_sequences(sequences, maxlen=max_text_length)
#     return padded_sequences

# def predict_proba(texts):
#     processed_texts = [preprocess_text(text) for text in texts]
#     return model.predict(np.vstack(processed_texts))

def predict_proba(arr):
    list_tokenized_ex = tokenizer.texts_to_sequences(arr)
    Ex = pad_sequences(list_tokenized_ex, maxlen=max_text_length)
    pred=model.predict(Ex)
    returnable=[]
    for i in pred:
      temp=i[0]
      returnable.append(np.array([1-temp,temp]))
    return np.array(returnable)

def get_explanation(text, samples=4000, num_features=10):
    explainer = LimeTextExplainer(class_names=['Not Upheld', 'Upheld'])
    explanation = explainer.explain_instance(text, predict_proba, num_features=num_features, num_samples=samples)
    return explanation

# def plot_explanation(explanation):
#     fig, ax = plt.subplots()
#     explanation.as_pyplot_figure(label=0, ax=ax)
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     return buf

def plot_explanation(explanation):
    fig = explanation.as_pyplot_figure(label=1)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Streamlit app
st.image("header.png", caption="About the tool", use_column_width=True)


# Description
st.markdown("""
This advanced tool has been developed as part of the MSc in Statistics with Data Science program at the University of Edinburgh, 
School of Mathematics. It leverages a novel Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM) units and attention mechanisms, 
designed to achieve high generalization performance.

Using this tool, you can:
- Paste the text from "The Complaint" and "What happened" sections.
- Predict the probability of financial ombudsman decision to being 'Upheld' or 'Not Upheld'.
- Receive explanations and heatmap using the explainable AI-based LIME algorithm.
- View an importance chart of the most significant words for predicting both classes.
""")

text_input = st.text_area("Enter your text here:")

n_samples = st.text_input('Number of samples to generate for LIME explainer:', value=5000)

n_features = st.text_input('Number of features (important words) to plot:', value=10)

if st.button("Explain"):
    if text_input:
        with st.spinner(f"Please wait {int(n_samples)//1000} seconds ...."):
            # Get LIME explanation
            text_processed = clean_spacy_text(text_input)
            explanation = get_explanation(text_processed, int(n_samples), int(n_features))
            
            # Display explainer HTML object
            components.html(explanation.as_html(), height=800)

    else:
        st.warning("Please enter some text to analyze.")
