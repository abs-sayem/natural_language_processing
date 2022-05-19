# Import the dependents
import sys
import numpy
import keras
from model import *
from keras.utils import np_utils

# Read text file
file = open("C:/Users/AbsSayem/.vscode/nlp/text_generation_using_lstm/pos.txt")
text = file.read()
text = text.lower()
#print(text)

# Create the mapping: cahracter to integer
number_of_chars = sorted(list(text))
char_to_int = dict((c,i) for i,c in enumerate(number_of_chars))
int_to_char = dict((i,c) for i,c in enumerate(number_of_chars))

# Check total characters
total_chars = len(text)
total_vocab = len(number_of_chars)

# Prepare the dataset pairs encoded as integers
dataX = []
dataY = []
for i in range(0, total_chars, 1):
    sequence_in = text[i:-1]
    sequence_out = text[-1]
    dataX.append([char_to_int[char] for char in sequence_in])
    dataY.append(char_to_int[sequence_out])
number_of_patterns = len(dataX)

X = numpy.reshape(dataX,number_of_patterns)     #Transform the list into sequence
X = X/total_vocab      #Rescale the data
y = np_utils.to_categorical(dataY)

#Fit the model into data
history = model.fit(X,y,epochs=10,bs=32)