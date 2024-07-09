import string
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

feedback = {
        'who makes chatbots': [(2, 0.), (0, 1.), (1, 1.), (0, 1.)],
        'about page': [(0, 1.)]
}
########################################
example_doc = docs[1]

# Splitting a text into words (while removing punctuation, etc.)
example_doc_tokenized = TOKENIZER.tokenize( example_doc.translate(REMOVE_PUNCTUATION_TABLE))

# Strip words of their plural forms, formatting, etc.
example_doc_tokenized_and_stemmed = [STEMMER.stem(token) for token in example_doc_tokenized]

# Transform a user query into a list of terms:
def tokenize_and_stem(s):
    return [STEMMER.stem(t) for t
            in TOKENIZER.tokenize(s.translate(REMOVE_PUNCTUATION_TABLE))]

# Gather vocabulary across the entire collection of documents by calling the fit method
vectorizer = TfidfVectorizer(tokenizer = tokenize_and_stem, stop_words = 'english')
vectorizer.fit(docs)


# Calculate cosine distance between the query vector and all vectors for all documents  in collection
query = 'contact email to chat to martin'
query_vector = vectorizer.transform([query]).toarray()
doc_vectors = vectorizer.transform(docs).toarray()
similarity = cosine_similarity(vectorizer.transform(['who makes chatbots']).toarray(), doc_vectors)

# Rank documents by their scores (elements of ranks are indexes of docs)
ranks= (-similarity).argsort(axis=None)
most_relevant_doc = docs[ranks[0]]
print(most_relevant_doc)
