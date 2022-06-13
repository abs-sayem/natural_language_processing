# text_generator_using_lstm-rnn
In this note book we will learn character generation using LSTM-RNN. We will follow the corresponding steps for character generation-
> 1. Load the Data
> 2. Required Preprocessing
> 3. Mapping Unique Characters and Summarize the Loaded Data
> 4. Split the Dataset
> 5. Normalize the Data
> 6. Define LSTM Model
> 7. Define the Checkpoint
> 8. Load the Network and Generate Text
#### Load the Text and Convert to Lower Case
After importing the libraries we will load the text data in reading mode and the encoding format will be UTF-8. After that we will make the whole dataset lower so that we can avoid the redundancy of words.
#### Mapping the Unique Characters
Mapping the unique characters helps us to understand the text better. Here, we can see the unique charecters in out text and we can then remove the unnecessary characters from the data. We first seperate the whole text into charecters and map them with a corresponding integer value. This will helps further to generate characters and convert to character.
#### Summarize the Loaded Data
In our text the total number of characters are 13636 and total vocabs(unique characters) are 41. These vocabs are being mapped earlier. We will use this characters for generating the corresponding character(vocab) further.
#### Split the Dataset
Now, we will split the whole dataset(text) into training and testing portions. dataX is for training and dataY is for validation. sequence_length will be used as first input. For this we will use first 100 characters as input for the first time. Using this input our model will predict next vocab(character) and from the updated character list the last 100 characters, including the last predicted character, will be used as next input to predict characters further. This will continue as much characters you wish to be predicted.
#### Normalize the Data
Before fit the model to data, we need to normalize our data for better fitting. Normalization scale the data between a specific range. It can be 0 to 1 0r -1 to 1. Normalization helps to decrease calculation and improve overfitting problem. It also helps a model to become genelarize, means- fit to unknown and unseen data.
#### Define the Model
To learn and predict we need a model. Here we will use LSTM model. LSTM is a upgraded version of RNN. It is best for time series data. Since, our purpose is to predict a character based on previous characters and going though, LSTM will works well in this situation. We can also try any kind of RNN model instead.
We define a simple sequential LSTM model with two Hidden layer along with a dropout layer of each. The first layer has 256 nodes with a dropout of 20%. The 2nd layer also has ythe same configuration with same dropout. Then a dense layer is used as output layer. We use softmax classifier to predict characters. The input layer has sequence size input nodes. We randomly seed the input sequence. The Model can be different in its configuration.
#### Load the Network and Generate Text
Our proceedings are almost finished. At this time, we just load the model to fit it to data and it will generate corresponding characters according to the sequence.
#### [Noted]
> We experienced the model predicts best in this config with our dataset. The dataset is very small in size.
> That means, if the dataset becomes lager then we have to define a larger model.
> The model predicts best with a high epochs, say- 1000 epochs. As much as it trains and validates, it predicts better.
