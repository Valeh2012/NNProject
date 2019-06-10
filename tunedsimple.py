import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from sklearn.metrics import accuracy_score
import keras.layers as layers
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras import optimizers, regularizers
from collections import Counter
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Input, Embedding, BatchNormalization, LSTM, Dense, Concatenate, Dropout
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from random import shuffle
import json
from azureml.core import Run

from sklearn.model_selection import train_test_split

# let user feed in 2 parameters, the location of the data files (from datastore), and the regularization rate of the logistic regression model
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
args = parser.parse_args()

data_folder = args.data_folder
print('Data folder:', data_folder)
with open(os.path.join(data_folder,'News_Category_Dataset_v2.json')) as json_file:  
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

    
train_df=pd.DataFrame(k)
train_df['category']=C
train_df.columns=["sentence","category"]


# parameter of max word length
time_steps = 100
# Create datasets (Only take up to time_steps words for memory)
train_text = train_df['sentence'].tolist()
train_text = [' '.join(t.split()[0:time_steps]) for t in train_text]
train_text = np.array(train_text)
train_label = np.array(train_df['category'].tolist())

MAX_LENGTH = 500
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_text)
post_seq = tokenizer.texts_to_sequences(train_text)
post_seq_padded = pad_sequences(post_seq, maxlen=MAX_LENGTH)



run = Run.get_context()


inputs = Input(shape=(500, ))
embedding_layer = Embedding(len(tokenizer.word_counts),
                            16,input_length=500)(inputs)
word_embedding= BatchNormalization()(embedding_layer)

x = LSTM(128)(word_embedding)
x = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.05))(x)
x = Dropout(0.2)(x)
# x = Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.05))(x)
predictions = Dense(41, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
rmsprop = optimizers.RMSprop(lr=0.001, decay=0.95)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()


X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, train_label, test_size=0.05)

#filepath="weights.hdf5"
#checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
          shuffle=False, epochs=5)


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
run.log_image('Model accuracy', plt)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
run.log_image('Model loss', plt)


predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis=1)
test_acc = accuracy_score(y_test, predicted)
print(test_acc)
run.log("accuracy", test_acc)



