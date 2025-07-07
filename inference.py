import tensorflow as tf
from transformers import DistilBertTokenizer
import numpy as np

MODEL_PATH = "weights/crypto_sentiment_model.h5"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(text):
    tokens = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors="tf")
    return tokens["input_ids"], tokens["attention_mask"]

def predict(text):
    input_ids, attention_mask = preprocess(text)
    preds = model([input_ids, attention_mask], training=False)
    label = tf.argmax(preds, axis=1).numpy()[0]
    return ["Negative", "Neutral", "Positive"][label]

if __name__ == "__main__":
    print(predict("Bitcoin is going to the moon!"))
