# nlp_preprocessing
Explore all the ways of preprocessing techniques for Natural Language Processing(nlp). Largely, preprossing includes - tokenization, stemming, lemmatization, parts-of-speech tagging(pos-tag), name entity recognition(ner), punctuation and stopwords removing, regular expression(re) etc.
## Tokenization
#### Using Python "split()" function
Python split() method by default split the text by words.
```
# Spliting by words
text = """This is text for testing preprocessing steps for nlp. For nlp, preprocessing steps are very much important. Preprocessing steps improve the accuracy in percentage(%). If you don't follow preprocessing steps, you can't get the desired result. Preprocessing steps varies from purpose to purpose."""
print(text.split())     # This will by default split the text by words
```
```
  ['This', 'is', 'text', 'for', 'testing', 'preprocessing', 'steps', 'for', 'nlp.', 'For', 'nlp,', 'preprocessing', 'steps', 'are', 'very', 'much', 'important.', 'Preprocessing', 'steps', 'improve', 'the', 'accuracy', 'in', 'percentage(%).', 'If', 'you', "don't", 'follow', 'preprocessing', 'steps,', 'you', "can't", 'get', 'the', 'desired', 'result.', 'Preprocessing', 'steps', 'varies', 'from', 'purpose', 'to', 'purpose.']
```
For splitting text into sentences, we can use python split() method providing a separator (. / ? / !).
```
# Spliting by words
text = """This is text for testing preprocessing steps for nlp. For nlp, preprocessing steps are very much important. Preprocessing steps improve the accuracy in percentage(%). If you don't follow preprocessing steps, you can't get the desired result. Preprocessing steps varies from purpose to purpose."""
text.split('.')     # This will  split the text by dots(.)
```
```
  ['This is text for testing preprocessing steps for nlp',
   ' For nlp, preprocessing steps are very much important',
   ' Preprocessing steps improve the accuracy in percentage(%)',
   " If you don't follow preprocessing steps, you can't get the desired result",
   ' Preprocessing steps varies from purpose to purpose',
   '']
```
#### Using Regular Expression
Regular Expression (re) is a special character sequence that helps to find strings or sets of strings using these sequences.
```
# Importing regular expression
import re
```
re.findall() function finds all the words and stores in a list.
* [\w]+ find all alphanumeric character(letters, numbers) and underscore(_) until any other character in encountered.
```
# Word Tokenization
text = """This is text for testing preprocessing steps for nlp. For nlp, preprocessing steps are very much important. Preprocessing steps improve the accuracy in percentage(%). If you don't follow preprocessing steps, you can't get the desired result. Preprocessing steps varies from purpose to purpose."""
tokens = re.findall("[\w']+", text)
print(tokens)
```
```
  ['This', 'is', 'text', 'for', 'testing', 'preprocessing', 'steps', 'for', 'nlp', 'For', 'nlp', 'preprocessing', 'steps', 'are', 'very', 'much', 'important', 'Preprocessing', 'steps', 'improve', 'the', 'accuracy', 'in', 'percentage', 'If', 'you', "don't", 'follow', 'preprocessing', 'steps', 'you', "can't", 'get', 'the', 'desired', 'result', 'Preprocessing', 'steps', 'varies', 'from', 'purpose', 'to', 'purpose']
```
re.compile() split sentences as soon as any of the separator are found.
```
# Sentence Tokenization
# To perform sentence tokenization we can use "re.split()" method. We can pass multiple separators in it.
text = """This is text for testing preprocessing steps for nlp. For nlp, preprocessing steps are very much important. Preprocessing steps improve the accuracy in percentage(%). If you don't follow preprocessing steps, you can't get the desired result. Preprocessing steps varies from purpose to purpose."""
sentences = re.compile('[.!?]').split(text)
sentences
```
```
  ['This is text for testing preprocessing steps for nlp',
   ' For nlp, preprocessing steps are very much important',
   ' Preprocessing steps improve the accuracy in percentage(%)',
   " If you don't follow preprocessing steps, you can't get the desired result",
   ' Preprocessing steps varies from purpose to purpose',
   '']
```
#### Using NLTK
* NLTK contains a module named tokenize() which has two sub-categories - 1) Word Tokenize and 2) Sentence Tokenize
  * punkt is a nltk library tool for tokenizing text documents. When we use an old or a degraded version of nltk module we generally need to download the remaining data -
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')
```
# Download and Import NLTK
!pip install nltk
import nltk
nltk.download('punkt')    #punkt is a nltk library tool for tokenizing text documents.
```
To tokenize in nltk, we can use word_tokenize module from nltk.tokenize class.
```
# Word tokenization
from nltk.tokenize import word_tokenize
text = """This is text for testing preprocessing steps for nlp. For nlp, preprocessing steps are very much important. Preprocessing steps improve the accuracy in percentage(%). If you don't follow preprocessing steps, you can't get the desired result. Preprocessing steps varies from purpose to purpose."""
print(word_tokenize(text))
```
```
  ['This', 'is', 'text', 'for', 'testing', 'preprocessing', 'steps', 'for', 'nlp', '.', 'For', 'nlp', ',', 'preprocessing', 'steps', 'are', 'very', 'much', 'important', '.', 'Preprocessing', 'steps', 'improve', 'the', 'accuracy', 'in', 'percentage', '(', '%', ')', '.', 'If', 'you', 'do', "n't", 'follow', 'preprocessing', 'steps', ',', 'you', 'ca', "n't", 'get', 'the', 'desired', 'result', '.', 'Preprocessing', 'steps', 'varies', 'from', 'purpose', 'to', 'purpose', '.']
```
For sentence tokenization, nltk has sent_tokenize module in nltk.tokenize class.
```
# Sentence tokenization
from nltk.tokenize import sent_tokenize
text = """This is text for testing preprocessing steps for nlp. For nlp, preprocessing steps are very much important. Preprocessing steps improve the accuracy in percentage(%). If you don't follow preprocessing steps, you can't get the desired result. Preprocessing steps varies from purpose to purpose."""
sent_tokenize(text)
```
```
  ['This is text for testing preprocessing steps for nlp.',
   'For nlp, preprocessing steps are very much important.',
   'Preprocessing steps improve the accuracy in percentage(%).',
   "If you don't follow preprocessing steps, you can't get the desired result.",
   'Preprocessing steps varies from purpose to purpose.']
```
#### Using spaCy
spaCy is an open-source library for advance NLP. It supports over 49+ languages and provides high computation speed.
```
# Import spaCy
import spacy
spacy.cli.download("en_core_web_sm")
```
* For word tokenization, we use English module from spacy.lang.en class.
* We load the English tokenizer to nlp. nlp is used for linguistic annotation.
* Then applying the annotations to our text we create a processed doc.
* Then for every token from doc, we append them to a tokens list.
```
# Word Tokenization
from spacy.lang.en import English
# Load english tokenizer
nlp = English()     # nlp is used to craeate documents with linguistic annotations.
text = """This is text for testing preprocessing steps for nlp. For nlp, preprocessing steps are very much important. Preprocessing steps improve the accuracy in percentage(%). If you don't follow preprocessing steps, you can't get the desired result. Preprocessing steps varies from purpose to purpose."""
doc = nlp(text)   # Creates the document using nlp
print(doc)

# tokenizing the document and create a list of tokens
tokens = []
for token in doc:
  tokens.append(token.text)
print(tokens)
```
```
  This is text for testing preprocessing steps for nlp. For nlp, preprocessing steps are very much important. Preprocessing steps improve the accuracy in percentage(%). If you don't follow preprocessing steps, you can't get the desired result. Preprocessing steps varies from purpose to purpose.
  ['This', 'is', 'text', 'for', 'testing', 'preprocessing', 'steps', 'for', 'nlp', '.', 'For', 'nlp', ',', 'preprocessing', 'steps', 'are', 'very', 'much', 'important', '.', 'Preprocessing', 'steps', 'improve', 'the', 'accuracy', 'in', 'percentage(%', ')', '.', 'If', 'you', 'do', "n't", 'follow', 'preprocessing', 'steps', ',', 'you', 'ca', "n't", 'get', 'the', 'desired', 'result', '.', 'Preprocessing', 'steps', 'varies', 'from', 'purpose', 'to', 'purpose', '.']
```
* To sentence tokenize, we need to create a sentencizer pipeline component and them to a pipeline.
* Then, we will create a doc using nlp and from this document, we will append the sentences to a list.
```
# Sentence Tokenization
from spacy.lang.en import English

# Load english tokenizer
nlp = English()     # nlp is used to craeate documents with linguistic annotations.

# Create the pipeline 'sentencizer' component
sbd = nlp.create_pipe('sentencizer')

# Add the component to the pipeline
nlp.add_pipe(sbd)

text = """This is text for testing preprocessing steps for nlp. For nlp, preprocessing steps are very much important. Preprocessing steps improve the accuracy in percentage(%). If you don't follow preprocessing steps, you can't get the desired result. Preprocessing steps varies from purpose to purpose."""
doc = nlp(text)   # Creates the document using nlp

# create list of sentence tokens
sents_list = []
for sent in doc.sents:
    sents_list.append(sent.text)
    #sents_list = sent.string.strip()
sents_list
```
```
  ['This is text for testing preprocessing steps for nlp.',
   'For nlp, preprocessing steps are very much important.',
   'Preprocessing steps improve the accuracy in percentage(%).',
   "If you don't follow preprocessing steps, you can't get the desired result.",
   'Preprocessing steps varies from purpose to purpose.']
```
#### Using Keras
It is an open-source neural network library for python. Keras can run on top of tensorflow.
```
# Install Keras
!pip install keras
```
* To perform word tokenization using keras, we use the text_to_word_sequence method from the keras.preprocessing.text class.
* Keras lower the case of all alphabets before tokenizing them.
```
# Word Tokenization
from keras.preprocessing.text import text_to_word_sequence
text = """This is text for testing preprocessing steps for nlp. For nlp, preprocessing steps are very much important. Preprocessing steps improve the accuracy in percentage(%). If you don't follow preprocessing steps, you can't get the desired result. Preprocessing steps varies from purpose to purpose."""
word_tokens = text_to_word_sequence(text)
print(word_tokens)
```
```
  ['this', 'is', 'text', 'for', 'testing', 'preprocessing', 'steps', 'for', 'nlp', 'for', 'nlp', 'preprocessing', 'steps', 'are', 'very', 'much', 'important', 'preprocessing', 'steps', 'improve', 'the', 'accuracy', 'in', 'percentage', 'if', 'you', "don't", 'follow', 'preprocessing', 'steps', 'you', "can't", 'get', 'the', 'desired', 'result', 'preprocessing', 'steps', 'varies', 'from', 'purpose', 'to', 'purpose']
```
#### Using Gensim
Gensim is an open source library for unsupervised topic modeling and natural language processing.
It can automatically extract semantic topics from a given document.
```
# Install Gensim
!pip install gensim
```
We can use the gensim.utils class to import the tokenize method to perform word tokenization.
```
# Word tokenization
from gensim.utils import tokenize
text = """This is text for testing preprocessing steps for nlp. For nlp, preprocessing steps are very much important. Preprocessing steps improve the accuracy in percentage(%). If you don't follow preprocessing steps, you can't get the desired result. Preprocessing steps varies from purpose to purpose."""
tokens = list(tokenize(text))
print(tokens)
```
```
  ['This', 'is', 'text', 'for', 'testing', 'preprocessing', 'steps', 'for', 'nlp', 'For', 'nlp', 'preprocessing', 'steps', 'are', 'very', 'much', 'important', 'Preprocessing', 'steps', 'improve', 'the', 'accuracy', 'in', 'percentage', 'If', 'you', 'don', 't', 'follow', 'preprocessing', 'steps', 'you', 'can', 't', 'get', 'the', 'desired', 'result', 'Preprocessing', 'steps', 'varies', 'from', 'purpose', 'to', 'purpose']
```
To perform sentence tokenization, we can use the split_sentences method from the gensim.summarization.textcleaner class.
```
# Sentence Tokenization
from gensim.summarization.textcleaner import split_sentences
text = """This is text for testing preprocessing steps for nlp. For nlp, preprocessing steps are very much important. Preprocessing steps improve the accuracy in percentage(%). If you don't follow preprocessing steps, you can't get the desired result. Preprocessing steps varies from purpose to purpose."""
sentences = split_sentences(text)
sentences
```
```
  ['This is text for testing preprocessing steps for nlp.',
   'For nlp, preprocessing steps are very much important.',
   'Preprocessing steps improve the accuracy in percentage(%).',
   "If you don't follow preprocessing steps, you can't get the desired result.",
   'Preprocessing steps varies from purpose to purpose.']
```
