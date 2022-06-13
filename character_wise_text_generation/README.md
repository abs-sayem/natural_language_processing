# text_generation_using_lstm_with_keras
A generative model for text (character-by-character) using LSTM recurrent neural networks in Python with Keras.
##### Here, we will discover how to create a generative model for text, character-by-character using LSTM recurrent neural networks in Python with Keras.

Here, we will learn--
1.   the text sequence problem and how to frame it to rnn generative model
2.   to develop a lstm model and generate text sequences for the problem.

[NB] LSTM recurrent network can be slow to train and it is highly recommended to train on GPU. We will use google colab here.

##### Follow the steps ---
```
1. Download and Import the dependencies
2. Upload the text file
3. Run the 'train_the_model.py' file
4. Run the 'generate_text.py' file
```
## Develop a LSTM model
#### *Import the libraries ---*
```
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
```
#### *Load the text file ---*
Here, we will open and read a text file for training. We need to make whole text in lowercase to reduce vocubulary redundancy.
```
file = open("nlp_preprocessing.txt")
text = file.read()
text = text.lower()
#print(text)
```
#### *Look up the character and vocabulary set---*
To model the data we must preprocess the dataset-
*   convert the characters into integers - because we cannot model the characters directly, first create a set of all the distinct characters in the text and then map them all to a unique integer.  This is our look-up table and we'll use it further to map our text to 1d matrix.
*   define the training data - pass
```
# create the mapping of character to integer
number_of_chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(number_of_chars))
int_to_char = dict((i, c) for i, c in enumerate(number_of_chars) )
#print(chars)
print(char_to_int)
# pass
total_chars = len(text)
total_vocab = len(number_of_chars)
print("Total Characters: ", total_chars)
print("Total Vocabulary: ", total_vocab)
```
#### *Prepare the dataset pairs encoded as integers ---*
* Our text has around 2500 total characters and when converted to lower case there is only 36 unique characters in the vocabulary. 
* To define the training data for the network, we'll split the text up into subsequences with a fixed length of 50 chatacters. 
* As we split up the text into these sequences, we convert the characters to integers using our lookup table we prepared earlier.
```
# Prepare the dataset pairs encoded as integers
sequence_length = 50
dataX = []
dataY = []
for i in range(0, total_chars - sequence_length, 1):
  sequence_in = text[i : i + sequence_length]
  sequence_out = text[i + sequence_length]
  dataX.append([char_to_int[char] for char in sequence_in])
  dataY.append(char_to_int[sequence_out])
number_of_patterns = len(dataX)
print("Total Patterns: ", number_of_patterns)
```
#### *Now we have three steps to do ---*
1. transform the list of input sequences into the form [samples, time steps, features] expected by a lstm network.
2. rescale the integers to the range 0-to-1 - it makes the patterns easier to learn by the LSTM network, uses the sigmoid activation function by default.
3. convert the output patterns into a one hot encoding.
```
# Step-1: transform the list input sequence
X = numpy.reshape(dataX, (number_of_patterns, sequence_length, 1))
#Step-2: rescale the input
X = X / float(total_vocab)
#Step-3: convert the output patterns into one-hot-encodding
y = np_utils.to_categorical(dataY)
```
#### *Define the LSTM Model ---*
Here, we define a single hidden lstm model with -
* 256 memory units (nodes).
* uses dropout with a probability of 20
* uses softmax activation function in the output layer.
    * output layer is a danse layer
    * softmax will predict for each of the 36 characters between 0 and 1
* uses adam optimizer - to optimize the loss
    * the problem is really a single character classification problem with 36 classes
    * to optimize the log loss (cross entropy) adam is used for speed
```
# Define the model---
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```
There is no test dataset here. So, we will model whole text to learn the probability of each character in a sequence. Here, we will give priority to learning (memorization).
The lstm network is slow to train. For this, we will use model checkpointing to record the improvement in loss at the end of the epoch.
#### *Define the Checkpoint ---*
```
# Define the checkpoint ---
filepath = "improved-weights-{epoch:02d}-{loss:.4f}-256x512x512x256.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
```
#### *Fit the Model to Data ---*
Now we can fit our data to the model. Here, we use 10 epochs with a batch size of 64 patterns. The more epochs you try the more the model gives precise word.

[NB] 10 epochs doesn't predict well. For better predicting more than 100 epochs are required. The more epochs you try the more well prediction will happen.
```
# Fit the model to data
history = model.fit(X, y, epochs=10, batch_size=64, callbacks=callbacks_list)
```
```
    Epoch 1/10
    36/37 [============================>.] - ETA: 0s - loss: 3.1712
    Epoch 00001: loss improved from inf to 3.17069, saving model to improved-weights-01-3.1707-256x512x512x256.hdf5
    37/37 [==============================] - 3s 23ms/step - loss: 3.1707
    Epoch 2/10
    36/37 [============================>.] - ETA: 0s - loss: 3.0535
    Epoch 00002: loss improved from 3.17069 to 3.05234, saving model to improved-weights-02-3.0523-256x512x512x256.hdf5
    37/37 [==============================] - 1s 21ms/step - loss: 3.0523
    Epoch 3/10
    37/37 [==============================] - ETA: 0s - loss: 3.0515
    Epoch 00003: loss improved from 3.05234 to 3.05155, saving model to improved-weights-03-3.0515-256x512x512x256.hdf5
    37/37 [==============================] - 1s 20ms/step - loss: 3.0515
    Epoch 4/10
    37/37 [==============================] - ETA: 0s - loss: 3.0342
    Epoch 00004: loss improved from 3.05155 to 3.03415, saving model to improved-weights-04-3.0342-256x512x512x256.hdf5
    37/37 [==============================] - 1s 18ms/step - loss: 3.0342
    Epoch 5/10
    34/37 [==========================>...] - ETA: 0s - loss: 3.0405
    Epoch 00005: loss improved from 3.03415 to 3.03202, saving model to improved-weights-05-3.0320-256x512x512x256.hdf5
    37/37 [==============================] - 1s 17ms/step - loss: 3.0320
    Epoch 6/10
    34/37 [==========================>...] - ETA: 0s - loss: 3.0230
    Epoch 00006: loss improved from 3.03202 to 3.03197, saving model to improved-weights-06-3.0320-256x512x512x256.hdf5
    37/37 [==============================] - 1s 18ms/step - loss: 3.0320
    Epoch 7/10
    35/37 [===========================>..] - ETA: 0s - loss: 3.0254
    Epoch 00007: loss improved from 3.03197 to 3.03042, saving model to improved-weights-07-3.0304-256x512x512x256.hdf5
    37/37 [==============================] - 1s 19ms/step - loss: 3.0304
    Epoch 8/10
    37/37 [==============================] - ETA: 0s - loss: 3.0291
    Epoch 00008: loss improved from 3.03042 to 3.02910, saving model to improved-weights-08-3.0291-256x512x512x256.hdf5
    37/37 [==============================] - 1s 18ms/step - loss: 3.0291
    Epoch 9/10
    35/37 [===========================>..] - ETA: 0s - loss: 3.0316
    Epoch 00009: loss did not improve from 3.02910
    37/37 [==============================] - 1s 17ms/step - loss: 3.0314
    Epoch 10/10
    36/37 [============================>.] - ETA: 0s - loss: 3.0275
    Epoch 00010: loss improved from 3.02910 to 3.02770, saving model to improved-weights-10-3.0277-256x512x512x256.hdf5
    37/37 [==============================] - 1s 18ms/step - loss: 3.0277
```
#### *Visualizing the loss history ---*
Loss history tells how many epochs you should use for training. In one point loss won't decrease or make a increasing tendency, the point would be the optimal point for the model.
```
#Summarize the loss history---
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
```
```
##### [Loss curve]
```
## Generating Text
Generating text using the trained LSTM network is straightforward - 

**1. Load the Data and Define the Network** - First, we load the data and define the network in exactly the same way, except the network weights are loaded from a checkpoint file and the network does not need to be trained.
```
# Load the network ---
filename = "improved-weights-02-3.0463.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
```
**2. Create a Reverse Mapping** - Also, when preparing the mapping of unique characters to integers, we must also create a reverse mapping that we can use to convert the integers back to characters so that we can understand the predictions.
```
int_to_char = dict((i, c) for i, c in enumerate(number_of_chars) )
```
**3. Make predictions** - Finally, we need to actually make predictions.

  *The simplest way to use the Keras LSTM model to make predictions is to first start with a seed sequence as input, generate the next character then update the seed sequence to add the generated character on the end and trim off the first character. This process is repeated for as long as we want to predict new characters (e.g. a sequence of 500 characters in length).*

  *We can pick a random input pattern as our seed sequence, then print generated characters.*
```
# Pick a random seed
start = numpy.random.randint(0,  len(dataX) - 1)
pattern = dataX[start]
print ("Seed: ")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# Generate Chatacters
for i in range(500):
  x = numpy.reshape(pattern, (1, len(pattern), 1))
  x = x/float(total_vocab)
  prediction = model.predict(x, verbose=0)
  index = numpy.argmax(prediction)
  result = int_to_char[index]
  seq_in = [int_to_char[value] for value in pattern]
  sys.stdout.write(result)
  pattern.append(index)
  pattern = pattern[1: len(pattern)]
print ("\nDone")
```
```
  Seed: 
  " uild the corpus by stripping all wikipedia markup  "
  from the articles, using gensim.
  you can read up on the wikicorpus class here.
  a second script then checks the corpus text file we just built.
  now, keep in mind that this large wikipedia dump file then resulted in a very large corpus file.
  given its enormous size, you may have dificulty reading the full file into memory at one time.
  this script, then, starts by reading 50 lines - which equates to 50 full articles - from the text file and outputting them to the terminal, after which you can press
  Done
```
## Larger LSTM Model
We can try to improve the quality of the generated text by creating a much larger network.

Defining hidden layers in LSTM is quite straightforward. We need to follow the basic steps ---
  1. a layer needs three parameters to connect with another layer (number of nodes. input_shape, return_sequence)
  2. the last layer has only a parameter (number of nodes)

Here, we define a LSTM model with two hidden layers. Both layer has 256 hidden nodes and a dropout of 20%.

**We can try different -**
  1. number of nodes
  2. dropout (can be removed)
  3. activation function
  4. loss function
  5. optimizer
##### LSTM Model with 2 HL
```
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```
##### LSTM Model with 3 HL
```
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```
## Ideas to Improve the Model
1. Removing punctuations and stopwords from the source text as well as input sequence
2. Increasing the number of training epochs
3. Tuning the dropout percentage
4. Adjusting the batch size, different optimizer/loss and activation function
