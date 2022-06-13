# Import the libraries
import numpy as np
import nltk
import string
import random
import warnings
# Load the corpus
#f = open('script.txt','r', errors='ignore')
f = open('nlp.txt','r', errors='ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower()
nltk.download('punkt')  # punkt is a pretrained tokenizer
nltk.download('wordnet')  # wordnet is like a dictionary
sent_tokens = nltk.sent_tokenize(raw_doc) # convert the txt file to a list of sentrences
word_tokens = nltk.word_tokenize(raw_doc) # # convert the txt file to a list of words
# Check the Corpus
for i in range(len(sent_tokens)):
  print(i, ":", sent_tokens[i])
# Preprocess the text
lemmer = nltk.stem.WordNetLemmatizer()
# WordNet is a sementically-oriented english dictionary included in NLTK. It removes all the punctution and also do lemmatization.
def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]
remove_punkt_dict = dict((ord(punkt), None) for punkt in string.punctuation)
def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punkt_dict)))
# Save them as pickle file
import pickle
with  open('qaa_text.pickle', 'wb') as f:
  pickle.dump(LemTokens, f)
# Define the greeting function
Greet_Inputs = ("hello","hi","hey","greetings","hello there","whats up")
Greet_Responses = ("Hi","Hi there","Hey","Hello","hello there","I'm glad! You are talking to me.")
def greet(sentence):
  for word in sentence.split():
    if word.lower() in Greet_Inputs:
      return random.choice(Greet_Responses)
# Feature Extruction
from sklearn.feature_extraction.text import TfidfVectorizer # Tf counts - how many times a word occured and idf counts - how rare a word is.
# tf: term frequency (the frequency of occurance of words) and idf: inverse document frequency (how rare the frequency of occurance of a word in the corpus)
from sklearn.metrics.pairwise import cosine_similarity
# Define the text generation function
def response(user_response):
  bot_response = ""
  tfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
  tfidf = tfidfVec.fit_transform(sent_tokens)
  values = cosine_similarity(tfidf[-1], tfidf)
  index = values.argsort()[0][-2]
  flat = values.flatten()
  flat.sort()
  req_tfidf = flat[-2]
  if(req_tfidf == 0):
    bot_response = bot_response + "I'm sorry! I don't understand."
    return bot_response
  else:
    bot_response = bot_response + sent_tokens[index]
    return bot_response
 # Start the chat
flag = True
print("Bot  : My name is ASL_Bot. Lets have a conversation. If you want to exit, just type bye. Don't thanks me.\n")
while(flag==True):
  user_response = input("User: ")
  user_response = user_response.lower()
  if(user_response != 'bye'):
    if(user_response == 'thanks' or user_response == 'thank you'):
      flag = False
      print("Bot  : You are welcome.\n")
    else:
      if(greet(user_response)!=None):
        print("Bot  : " + greet(user_response) + "\n")
      else:
        sent_tokens.append(user_response)
        word_tokens = word_tokens + nltk.word_tokenize(user_response)
        final_words = list(set(word_tokens))
        print("Bot  : ", end="")
        warnings.filterwarnings("ignore")
        val = response(user_response)
        print(val + "\n")
        sent_tokens.remove(user_response)
  else:
    flag = False
    print("Bot  : Good Bye. Have a nice day.")
