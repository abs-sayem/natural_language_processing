# question-answering-app
A simple chatbot that import a text file and answer any question related to the text file. 
### Import the Libraries
  **NumPy** or **Numerical Python** is used for various array operations. You can import numpy using `import numpy as np`<br>
  **NLTK** provides text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning also has over 50 corpora and lexical resources such as wordnet. To use its libraries import nltk by `import nltk` command.
```
import numpy as np
import nltk
import string
import random
import warnings
```
### Importing and Reading the Corpus
```
#f = open('script.txt','r', errors='ignore')
f = open('nlp.txt','r', errors='ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower()
nltk.download('punkt')  # punkt is a pretrained tokenizer
nltk.download('wordnet')  # wordnet is like a dictionary
sent_tokens = nltk.sent_tokenize(raw_doc) # convert the txt file to a list of sentrences
word_tokens = nltk.word_tokenize(raw_doc) # # convert the txt file to a list of words
```
### Check the Corpus: how it look like
```
#sent_tokens
#print("Sent_Tokens: " , sent_tokens)
#sent_tokens
for i in range(len(sent_tokens)):
  print(i, ":", sent_tokens[i])
```
```
 0 : one of the first things required for natural language processing (nlp) tasks is a corpus.
 1 : in linguistics and nlp, corpus refers to a collection of texts.
 2 : such collections may be formed of a single language of texts, or can span multiple languages - there are numerous reasons for which multilingual corpora, the plural of corpus, may be useful.
 3 : corpora may also consist of themed texts, like- historical, biblical, etc.
 4 : corpora are generally solely used for statistical linguistic analysis and hypothesis testing.
 5 : the good thing is that the internet is filled with text, and in many cases this text is collected and well oganized, even if it requires some finessing into a more usable, precisely-defined format.
 6 : wikipedia, in particular, is a rich source of well-organized textual data.
 7 : it's also a vast collection of knowledge, and the unhampered mind can dream up all sorts of uses for just such a body of text.
 8 : what we will do here is build a corpus from the set of english wikipedia articles, which is freely and conveniently available online.
 9 : in order to easily build a text corpus void of the wikipedia article markup, we will use gensim, a topic modeling library for python.
 10 : a wikipedia dump file is also required for this procedure, quite obviously.
 11 : the latest such files can be found here.
 12 : i wrote a simple python script to build the corpus by stripping all wikipedia markup from the articles, using gensim.
 13 : you can read up on the wikicorpus class here.
 14 : a second script then checks the corpus text file we just built.
 15 : now, keep in mind that this large wikipedia dump file then resulted in a very large corpus file.
 16 : given its enormous size, you may have dificulty reading the full file into memory at one time.
 17 : this script, then, starts by reading 50 lines - which equates to 50 full articles - from the text file and outputting them to the terminal, after which you can press a key to output another 50, or type 'stop' to quit.
 18 : if you do stop, the script then proceeds to load the entire file into memory.
 19 : which could be a problem for you.
 20 : you can, however, verify the text by batches of lines, in order to satisfy your curiousity that something good happened as a result of running the first script.
 21 : if you are planning on working on such a large text file, you may need some workarounds for its large size in comparison to your machine's memory.
```
### Text Preprocessing
```
lemmer = nltk.stem.WordNetLemmatizer()
# WordNet is a sementically-oriented english dictionary included in NLTK. It removes all the punctution and also do lemmatization.
def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]
remove_punkt_dict = dict((ord(punkt), None) for punkt in string.punctuation)
def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punkt_dict)))
```
### Save as Pickle
```
import pickle
with  open('qaa_text.pickle', 'wb') as f:
  pickle.dump(LemTokens, f)
```
### Initializing with Greetings
```
Greet_Inputs = ("hello","hi","hey","greetings","hello there","whats up")
Greet_Responses = ("Hi","Hi there","Hey","Hello","hello there","I'm glad! You are talking to me.")
def greet(sentence):
  for word in sentence.split():
    if word.lower() in Greet_Inputs:
      return random.choice(Greet_Responses)
```
### Response Generation
```
# Feature Extruction
from sklearn.feature_extraction.text import TfidfVectorizer # Tf counts - how many times a word occured and idf counts - how rare a word is.
# tf: term frequency (the frequency of occurance of words) and idf: inverse document frequency (how rare the frequency of occurance of a word in the corpus)
from sklearn.metrics.pairwise import cosine_similarity

# Generate the response
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
```
### Defining Conversation Protocol: Start the Chat
```
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
```
```
 Bot  : My name is ASL_Bot. Lets have a conversation. If you want to exit, just type bye. Don't thanks me.

 User: hello
 Bot  : hello there

 User: what does corpus refer to?
 Bot  : such collections may be formed of a single language of texts, or can span multiple languages - there are numerous reasons for which multilingual corpora, the plural of corpus, may be useful.

 User: what does corpus refer to
 Bot  : such collections may be formed of a single language of texts, or can span multiple languages - there are numerous reasons for which multilingual corpora, the plural of corpus, may be useful.

 User: what is required for this procedure?
 Bot  : a wikipedia dump file is also required for this procedure, quite obviously.

 User: what is the first things required for nlp
 Bot  : one of the first things required for natural language processing (nlp) tasks is a corpus.

 User: how to easily build a text corpus
 Bot  : in order to easily build a text corpus void of the wikipedia article markup, we will use gensim, a topic modeling library for python.

 User: thanks
 Bot  : You are welcome.
```
