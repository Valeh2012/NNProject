from azureml.core import Run
import pandas as pd
import json

with open('News_Category_Dataset_v2.json') as json_file:  
    data = json.load(json_file)
   
sample = [ x for x in data if x['category'] in ['HEALTHY LIVING','WORLD NEWS','COMEDY','SPORTS','BLACK VOICES'] ]

from random import shuffle
shuffle(sample)

train = sample[:20000]
validation = sample[20000:23000]
test = sample[23000:]

d={'HEALTHY LIVING':1,'WORLD NEWS':2,'COMEDY':3,'SPORTS':4,'BLACK VOICES':0}

L=[]
k=[]
C=[]
for item in train:
    c=d[item['category']]
    C.append(c)
    exp=[item['headline'],item['short_description']]
    EXP=item['headline']+item['short_description']
    L.append(exp)
    k.append(EXP)

T=[]
C_T=[ ]
for item in test:
    c=d[item['category']]
    EXP=item['headline']+item['short_description']
    T.append(EXP)
    C_T.append(c)

train_df = pd.DataFrame(k)
test_df = pd.DataFrame(T)

train_df['category']=C
test_df['category']=C_T

train_df.columns=["sentence","category"]
test_df.columns=["sentence","category"]

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import keras.layers as layers

from collections import Counter
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Input, Embedding, BatchNormalization, LSTM, Dense, Concatenate,Activation
from keras.models import Model

# from keras.utils import plot_mode
from keras.utils import to_categorical

# Reduce TensorFlow logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

# Instantiate the elmo model
elmo_module = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

# Initialize session
sess = tf.Session()
K.set_session(sess)

K.set_learning_phase(1)

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

# parameter of max word length
time_steps = 100
# building vocabulary from dataset
def build_vocabulary(sentence_list):
    unique_words = " ".join(sentence_list).strip().split()
    word_count = Counter(unique_words).most_common()
    vocabulary = {}
    for word, _ in word_count:
        vocabulary[word] = len(vocabulary)        

    return vocabulary


# Get vocabulary vectors from document list
# Vocabulary vector, Unknown word is 1 and padding is 0
# INPUT: raw sentence list
# OUTPUT: vocabulary vectors list
def get_voc_vec(document_list, vocabulary):    
    voc_ind_sentence_list = []
    for document in document_list:
        voc_idx_sentence = []
        word_list = document.split()
        
        for w in range(time_steps):
            if w < len(word_list):
                # pickup vocabulary id and convert unknown word into 1
                voc_idx_sentence.append(vocabulary.get(word_list[w], -1) + 2)
            else:
                # padding with 0
                voc_idx_sentence.append(0)
            
        voc_ind_sentence_list.append(voc_idx_sentence)
        
    return np.array(voc_ind_sentence_list)


vocabulary = build_vocabulary(train_df["sentence"])

# mini-batch generator
def batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("batch_size", batch_size)
    print("num_batches_per_epoch", num_batches_per_epoch)

    def data_generator():
        data_size = len(data)

        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                                
                X_voc = get_voc_vec(shuffled_data[start_index: end_index], vocabulary)
                                
                sentence_split_list = []
                sentence_split_length_list = []
            
                for sentence in shuffled_data[start_index: end_index]:    
                    sentence_split = sentence.split()
                    sentence_split_length = len(sentence_split)
                    sentence_split += ["NaN"] * (time_steps - sentence_split_length)
                    
                    sentence_split_list.append((" ").join(sentence_split))
                    sentence_split_length_list.append(sentence_split_length)
        
                X_elmo = np.array(sentence_split_list)

                X =  X_elmo
                y = shuffled_labels[start_index: end_index]
                
                yield X, y

    return num_batches_per_epoch, data_generator()

# embed elmo method
def make_elmo_embedding(x):
    embeddings = elmo_module(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]
    
    return embeddings

run = Run.get_context()

# elmo embedding dimension
elmo_dim = 1024

# Input Layers
#word_input = Input(shape=(None, ), dtype='int32')  # (batch_size, sent_length)
elmo_input = Input(shape=(None, ), dtype=tf.string)  # (batch_size, sent_length, elmo_size)

# Hidden Layers
#word_embedding = Embedding(input_dim=len(vocabulary), output_dim=128, mask_zero=True)(word_input)
elmo_embedding = layers.Lambda(make_elmo_embedding, output_shape=(None, elmo_dim))(elmo_input)
#word_embedding = Concatenate()([word_embedding, elmo_embedding])
word_embedding = BatchNormalization()(elmo_embedding)
x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(word_embedding)

# Output Layer
m = Dense(units=5, activation='softmax')(x)
predict = Dense(units=1)(m)


model = Model(inputs=[elmo_input], outputs=predict)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

model.summary()

#plot_model(model, to_file="model.png", show_shapes=True)

# Create datasets (Only take up to time_steps words for memory)
train_text = train_df['sentence'].tolist()
train_text = [' '.join(t.split()[0:time_steps]) for t in train_text]
train_text = np.array(train_text)
train_label = np.array(train_df['category'].tolist())

test_text = test_df['sentence'].tolist()
test_text = [' '.join(t.split()[0:time_steps]) for t in test_text]
test_text = np.array(test_text)
test_label = np.array(test_df['category'].tolist())

# mini-batch size
batch_size = 32

train_steps, train_batches = batch_iter(train_text,
                                        np.array(train_df["category"]),
                                        batch_size)
valid_steps, valid_batches = batch_iter(test_text,
                                        np.array(test_df["category"]),
                                        batch_size)

logfile_path = './log'
tb_cb = TensorBoard(log_dir=logfile_path, histogram_freq=0)

history = model.fit_generator(train_batches, train_steps,
                              epochs=5, 
                              validation_data=valid_batches,
                              validation_steps=valid_steps,
                              callbacks=[tb_cb])


os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/model.pkl')
