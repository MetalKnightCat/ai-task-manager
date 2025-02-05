import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Sample training data (task text, priority)
training_data = [
    ("Finish project report urgent", 3),
    ("Buy groceries milk eggs", 1),
    ("Prepare presentation for client meeting", 3),
    ("Read new research paper", 2),
    ("Exercise 30 minutes", 2)
]

# Preprocess data
tokenizer = Tokenizer(num_words=1000)
texts = [item[0] for item in training_data]
labels = [item[1] - 1 for item in training_data]  # Convert to 0-based
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to make them uniform in length
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Build model
model = Sequential([
    Embedding(1000, 16, input_length=max_length),
    LSTM(32),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(np.array(padded_sequences), np.array(labels), epochs=50)

def predict_priority(task_text):
    seq = tokenizer.texts_to_sequences([task_text])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post')
    prediction = model.predict(padded_seq)
    return np.argmax(prediction) + 1  # Convert back to 1-3 scale