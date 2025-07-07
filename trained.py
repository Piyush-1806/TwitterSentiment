import tensorflow as tf
import pandas as pd
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased")

df = pd.read_csv("tweets.csv")  # CSV with 'text' and 'sentiment' columns
X = df["text"].tolist()
y = LabelEncoder().fit_transform(df["sentiment"].tolist())

input_ids = []
attention_masks = []

for tweet in X:
    encoded = tokenizer(tweet, truncation=True, padding='max_length', max_length=128)
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])

X_train, X_test, y_train, y_test = train_test_split(
    (tf.constant(input_ids), tf.constant(attention_masks)),
    tf.constant(y), test_size=0.1, random_state=42
)

# Define model
input_ids_layer = tf.keras.Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask_layer = tf.keras.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
embedding = bert(input_ids_layer, attention_mask=attention_mask_layer)[0][:, 0, :]
x = tf.keras.layers.Dense(64, activation='relu')(embedding)
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=X_train, y=y_train, validation_split=0.1, epochs=2, batch_size=32)

model.save("weights/crypto_sentiment_model.h5")
