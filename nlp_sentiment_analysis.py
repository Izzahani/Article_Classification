# %% Import packages
import pandas as pd
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
import datetime
import pickle, json

# %% Step 1) Loading data
df = pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')

# Features --> X
text = df['text']

# Target --> Y
category = df['category']

# %% Step 2) Data Inspection

# to check NaN
df.isna().sum()

# %%
df.info()

# %%
df.describe()

# %% 
df.duplicated().sum() # 99 duplicated data

# %%
print(text[11])


# %% Step 3) Data Cleaning
# to remove anything other than alphabets
temp = []

for index, txt in enumerate(text):
    text[index] = re.sub(r'[^a-zA-Z]', ' ', text[index]).lower()
    temp.append(len(text[index].split()))

print('Average Length of Words per Sentences: ' + str(np.mean(temp))) # to find the average length of the sentences in every sentences # around 300
print('Middle Value Length of Words per Sentences: '+ str(np.median(temp))) #335
# %%
print(text[11])

# %% Step 4) Features Selection
df1 = pd.concat([text,category], axis=1)
df1 = df1.drop_duplicates()

text = df1['text']
category = df1['category']

# %% Step 5) Data Pre-processing
# for features
# use tokenizers
num_words = 5000 
oov_token = 'OOV'

tokenizer = Tokenizer(num_words=num_words, oov_token = oov_token)
tokenizer.fit_on_texts(text)

word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(text)

# %%
train_sequences = pad_sequences(train_sequences, maxlen=100, padding='post', truncating='post' )

# %%
# for target
print(category.unique().sum())

# convert target into numerical
ohe = OneHotEncoder(sparse=False)
train_category = ohe.fit_transform(category[::, None])

# %% Model Development
# train test split
train_sequences = np.expand_dims(train_sequences, -1)

X_train, X_test, y_train, y_test = train_test_split(train_sequences, train_category)
# %% Model Evaluation
embedding_size = 64
model = Sequential()

# use embedding as input layer
model.add(Embedding(num_words, embedding_size)) # num_words = 5000
model.add(LSTM(128,return_sequences = True))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128,return_sequences = True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64,return_sequences = True)))
model.add(LSTM(64))
model.add(Dense(5, activation = 'softmax')) # 5 outputs
model.summary()

# %%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc') 

# %%
logs_path = os.path.join(os.getcwd(),'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = TensorBoard(log_dir=logs_path)
early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, epochs=5, callbacks=[tensorboard_callback, early_stop_callback], validation_data=(X_test, y_test))

# %%
y_pred = model.predict(X_test)

# %%
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test,axis=1)

print(classification_report(y_true, y_pred))

# %% Model Saving

token_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    json.dump(token_json, f)

# %% save ohe

with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe,f)

# %% save deep learning model

model.save('model sentiment analysis.h5')