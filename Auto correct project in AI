# Import required libraries
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.models import Sequential
from nltk.corpus import words
import numpy as np
import re

# Step 1: Data Collection and Preprocessing
# Download a set of English words from nltk
import nltk
nltk.download('words')
word_list = words.words()

# Function to introduce common spelling errors (typos) to generate artificial dataset
def introduce_typo(word):
    if len(word) < 4:
        return word  # avoid modifying short words
    # Replace random character
    word = list(word)
    rand_index = np.random.randint(1, len(word) - 1)
    word[rand_index] = chr(np.random.randint(97, 122))  # random lowercase char
    return ''.join(word)

# Create dataset with correct and incorrect words
data = [(word, introduce_typo(word)) for word in word_list[:10000]]  # sample data
X, y = zip(*data)

# Step 2: NLP Model Development
# Convert words to sequences (tokenize) for model input
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
y_seq = tokenizer.texts_to_sequences(y)

# Pad sequences for consistent input size
max_len = max([len(seq) for seq in X_seq])
X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=max_len, padding='post')
y_pad = tf.keras.preprocessing.sequence.pad_sequences(y_seq, maxlen=max_len, padding='post')

# Build NLP model (LSTM based)
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_len),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dense(128, activation='relu'),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')  # output layer for char-level prediction
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 3: Training and Optimization
# Train the model (here for simplicity, X_pad and y_pad are the same, but in real-world, they'd differ)
y_pad_shifted = np.expand_dims(y_pad, -1)  # expand for correct loss calculation
model.fit(X_pad, y_pad_shifted, epochs=5, batch_size=64, validation_split=0.2)

# Step 4: Real-time Implementation (Auto-correct Function)
# Function to predict corrected word
def auto_correct(input_word):
    seq = tokenizer.texts_to_sequences([input_word])
    seq_pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(seq_pad)
    pred_word = ''.join([tokenizer.index_word.get(np.argmax(char), '') for char in pred[0]])
    return re.sub(r'[^a-zA-Z]', '', pred_word)

# Example usage of auto-correct
misspelled_word = "helo"
corrected_word = auto_correct(misspelled_word)
print(f"Original: {misspelled_word}, Corrected: {corrected_word}")

# Step 5: Evaluation and Fine-tuning
# Evaluate the model on a test dataset (simple validation here)
test_loss, test_acc = model.evaluate(X_pad, y_pad_shifted)
print(f"Test Accuracy: {test_acc}")

# Step 6: Deployment and Continuous Improvement
# In production, you would wrap the model in a web API (e.g., using Flask) and monitor real-time feedback to improve.

