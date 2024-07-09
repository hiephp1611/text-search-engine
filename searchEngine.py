import string
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer

REMOVE_PUNCTUATION_TABLE= str.maketrans({x: None for x in string.punctuation})
TOKENIZER = TreebankWordTokenizer()
STEMMER = PorterStemmer()

#######################################
docs = [
    '''About us. We deliver Artificial Intelligence & Machine Learning
       solutions to solve business challenges.''',
    '''Contact information. Email [martin davtyan at filament dot ai]
       if you have any questions''',
    '''Filament Chat. A framework for building and maintaining a scalable
       chatbot capability''',
]

########################################
example_doc = docs[1]

# Splitting a text into words (while removing punctuation, etc.)
example_doc_tokenized = TOKENIZER.tokenize( example_doc.translate(REMOVE_PUNCTUATION_TABLE))

# Strip words of their plural forms, formatting, etc.
example_doc_tokenized_and_stemmed = [STEMMER.stem(token) for token in example_doc_tokenized]
print(example_doc_tokenized_and_stemmed)
# Transform a user query into a list of terms:
def tokenize_and_stem(s):
    return [STEMMER.stem(t) for t
            in TOKENIZER.tokenize(s.translate(REMOVE_PUNCTUATION_TABLE))]
