###
###  Simple Word Embedding 
###
import numpy as np
import os
import pandas as pd
import re
import keras.layers as layers

from collections import Counter
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Input, Embedding, BatchNormalization, LSTM, Dense, Concatenate
from keras.models import Model
from sklearn.externals  import joblib

import pandas as pd
import json
from random import shuffle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical


from azureml.core import Run

with open('News_Category_Dataset_v2.json') as json_file:  
    train_data = json.load(json_file)

    
shuffle(train_data)

a=[]
for i in train_data:
    a.append(i['category'])
b=set(a)

d = {}
for k,v in enumerate(b):
    d[v] = k
    
k=[]
C=[]
for item in train_data:
    c=d[item['category']]
    C.append(c)
    EXP=item['headline']+item['short_description']
    k.append(EXP)

train_df = pd.DataFrame(k)
train_df['category']=C
train_df.columns=["sentence","category"]

time_steps = 100

def build_vocabulary(sentence_list):
    unique_words = " ".join(sentence_list).strip().split()
    word_count = Counter(unique_words).most_common()
    vocabulary = {}
    for word, _ in word_count:
        vocabulary[word] = len(vocabulary)        

    return vocabulary

vocabulary = build_vocabulary(train_df["sentence"])

run = Run.get_context()

inputs = Input(shape=(500, ))
embedding_layer = Embedding(len(vocabulary),
                            128,input_length=500)(inputs)
word_embedding= BatchNormalization()(embedding_layer)

x = LSTM(128)(word_embedding)
x = Dense(64, activation='relu')(x)
predictions = Dense(41, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()



train_text = train_df['sentence'].tolist()
train_text = [' '.join(t.split()[0:time_steps]) for t in train_text]
train_text = np.array(train_text)
train_label = np.array(train_df['category'].tolist())

MAX_LENGTH = 500
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_text)
post_seq = tokenizer.texts_to_sequences(train_text)
post_seq_padded = pad_sequences(post_seq, maxlen=MAX_LENGTH)

X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, train_label, test_size=0.05)

filepath="weights.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
                        shuffle=False, epochs=10, callbacks=[checkpointer])

    
    
os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/model2.pkl')