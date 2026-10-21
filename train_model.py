
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Project settings
VOCAB_SIZE = 10000
MAX_LENGTH = 200
MODEL_DIR = "artifacts"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading IMDB movie review dataset...")

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

print("Preparing review sequences...")

x_train = pad_sequences(
    x_train, maxlen=MAX_LENGTH, padding="post", truncating="post"
)
x_test = pad_sequences(
    x_test, maxlen=MAX_LENGTH, padding="post", truncating="post"
)

print("Building LSTM model...")

model = Sequential([
    Embedding(VOCAB_SIZE, 64, mask_zero=True),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("Training the model...")

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

model.fit(
    x_train,
    y_train,
    epochs=8,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print("Evaluating model on test data...")

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

print(f"Test accuracy: {accuracy * 100:.2f}%")

print("Saving model...")

model.save(os.path.join(MODEL_DIR, "movie_review_lstm.keras"))

word_index = imdb.get_word_index()

with open(os.path.join(MODEL_DIR, "word_index.pkl"), "wb") as f:
    pickle.dump(word_index, f)

print("Training completed successfully!")
print("Model and word index saved in the artifacts folder.")
