import string
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
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






# Transform a user query into a list of terms:
def tokenize_and_stem(s):
    return [STEMMER.stem(t) for t
            in TOKENIZER.tokenize(s.translate(REMOVE_PUNCTUATION_TABLE))]
    

class Scorer():
    """ Scores documents for a search query based on tf-idf
        similarity and relevance feedback
        
    """
    
    def __init__(self, docs):
        """ Initialize a scorer with a collection of documents, fit a 
            vectorizer and list feature functions
        
        """
        self.docs = docs
        # Gather vocabulary across the entire collection of documents by calling the fit method
        # tokenzier Splitting a text into words (while removing punctuation, etc.)
        # stem Strip words of their plural forms, formatting, etc.
        # TfidfVectorizer Gather vocabulary across the entire collection of documents by calling the fit method
        
        self.vectorizer = TfidfVectorizer(tokenizer= tokenize_and_stem,
                                          stop_words= 'english')
        self.doc_tfidf = self.vectorizer.fit_transform(docs)
        
        self.features = [
            self._feature_tfidf,
            self._feature_positive_feedback,
        ]
        self.feature_weights = [
            1.,
            2.,
        ]
        
        self.feedback = {}
        
    def score(self, query):
        """ Generic scoring function: for a query output a numpy array
            of scores aligned with a document list we initialized the
            scorer with
        
        """
        feature_vectors = [feature(query) for feature
                           in self.features]
        
        feature_vectors_weighted = [feature * weight for feature, weight
                                    in zip(feature_vectors, self.feature_weights)]
        return np.sum(feature_vectors_weighted, axis= 0)
    
    def learn_feedback(self, feedback_dict):
        """ Learn feedback in a form of `query` -> (doc index, feedback value).
            In real life it would be an incremental procedure updating the
            feedback object.
        
        """
        self.feedback = feedback_dict
        
    def _feature_tfidf(self,query):
        """ TF-IDF feature. Return a numpy array of cosine similarities
            between TF-IDF vectors of documents and the query
        
        """
        query_vector = self.vectorizer.transform([query])
        # Calculate cosine distance between the query vector and all vectors for all documents  in collection
        similarity = cosine_similarity(query_vector, self.doc_tfidf)
        return similarity.ravel()
    
    def _feature_positive_feedback(self, query):
        """ Positive feedback feature. Search the feedback dict for a query
            similar to the given one, then assign documents positive values
            if there is positive feedback about them.
        
        """
        if not self.feedback:
            return np.zeros(len(self.docs))
        # Get feedback_query
        feedback_queries = list(self.feedback.keys())
        
        # Calculate cosine distance between the query vector and all vectors for all documents  in collection
        similarity = cosine_similarity(self.vectorizer.transform([query]),
                                       self.vectorizer.transform(feedback_queries))
        # Scale with a similarity between the original query and nearest neighbor query, and output a feature vector
        nn_similarity = np.max(similarity)
        
        nn_idx = np.argmax(similarity)
        # Look at all documents that received positive feedback
        pos_feedback_doc_idx = [idx for idx, feedback_value in 
                                self.feedback[feedback_queries[nn_idx]]
                                if feedback_value == 1.]
        counts = Counter(pos_feedback_doc_idx)
        feature_values = {
            doc_idx: nn_similarity * count / sum(counts.values())
            for doc_idx, count in counts.items()
        }
        return np.array([feature_values.get(doc_idx, 0.)
                         for doc_idx, _ in enumerate(self.docs)])
        
scorer = Scorer(docs)
scorer.learn_feedback(feedback)
scorer.feature_weights = [0.6, 0.4]
query = 'who is making chatbots information'
scorer.score(query)

print(docs[scorer.score(query).argmax()])