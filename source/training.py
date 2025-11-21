# Training sript
import os
import re
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

## Configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
data_path = "data/Tweets.csv"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# hyperparams
max_num_words = 30000
max_seq_len = 100
Embedding_dim = 100
batch_size = 64
epochs = 6
test_size = 0.15
val_size = 0.15


# Utilities
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load data
df = pd.read_csv(data_path)

# Inspecting cols names 
print(f"Columns: {df.columns.tolist()}")

# cleaning
df = df[['text', 'airline_sentiment']].dropna()
df['text'] = df['text'].astype(str).apply(clean_text)

## Enconding labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['airline_sentiment'])
print(f"Label classes: {list(le.classes_)}")

# Split train and test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    df['text'], df['label'], test_size=test_size, stratify=df['label'], random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=val_size/(1- test_size), stratify=y_train_full, random_state=SEED
)

# Tokenization
tokenizer = Tokenizer(num_words=max_num_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq   = tokenizer.texts_to_sequences(X_val)
X_test_seq  = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_len, padding='post', truncating='post')
X_val_pad   = pad_sequences(X_val_seq, maxlen=max_seq_len, padding='post', truncating='post')
X_test_pad  = pad_sequences(X_test_seq, maxlen=max_seq_len, padding='post', truncating='post')

# Save the tokenizer
with open(os.path.join(model_dir, "tokenizer.pickle"), "wb") as f:
    pickle.dump(tokenizer, f)

# Computing class weights ti mitigate imbalance
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)


## Build the model
num_classes = len(le.classes_)

model = Sequential() #To initialize the model
model.add(Input(shape=(max_seq_len,)))    #Input Layer
model.add(Embedding(input_dim=max_num_words, output_dim=Embedding_dim ))  #hiden
model.add(Bidirectional(LSTM(64, return_sequences=False, recurrent_dropout=0.15)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax')) #Output Layer

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.summary()


## Callbacks
checkpoint_path = os.path.join(model_dir, "lstm_sentiment.keras")
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_accuracy', save_best_only=True, verbose=1)
early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# Traininng
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint, early],
    class_weight = class_weight_dict
)

# Evaluation
test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=1)
print(f"Test accuracy: {test_acc:.4f}")

# Load the best model and predict
best_model = load_model(checkpoint_path)

y_pred_probs = best_model.predict(X_test_pad)
y_preds = np.argmax(y_pred_probs, axis=1)

print("Classification report:")
print(classification_report(y_test, y_preds, target_names=le.classes_))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_preds))

# Save label encoder mapping for inference
with open(os.path.join(model_dir, "label_encoder.pickle"), "wb") as f:
    pickle.dump(le, f)

print("Done. Models and tokenizer saved to:", model_dir)