import numpy as np,pandas as pd
from sklearn.base import BaseEstimator
from typing import Literal
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from umap import UMAP # umap stage is the botleneck, may try cuml.manifold.UMAP later
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired,MaximalMarginalRelevance,BaseRepresentation
from bertopic.backend import BaseEmbedder
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.cluster import BaseCluster
from nltk.util import everygrams



class TopicModelBase(BaseEstimator) :
    """BERTopic(...) Default"""
    def __init__(self,top_n_words=10) :
        self.top_n_words = top_n_words

    def _load_model(self) -> BERTopic :
        """Modify as per requirement"""
        return BERTopic(top_n_words=self.top_n_words)

    def fit_transform(self,X:pd.Series,y:pd.Series|None=None,*,embeddings:np.ndarray|None=None) -> pd.DataFrame :
        """ 
        INPUT : 
            X -- pandas.Series[str]
                Narratives 
            y -- pd.Series 
                Class labels
            embeddings -- numpy.ndarray
                Pre-computed document embeddings

        RETURN : 
            pd.DataFrame -- `get_topic_info()` output of the fitted model
                
        ATTRIBUTES :
            topic_model_ -- underlying fitted `BERTopic(...)` instance (with all its attributes & methods)
            vocabulary_ -- the underlying vocabulary

        Ref: https://maartengr.github.io/BERTopic/api/bertopic.html
        """ 
        X = X.to_list()
        y = pd.factorize(y,sort=True,use_na_sentinel=True)[0] if y is not None else y # sort=True handles already int labels
        self.topic_model_ = self._load_model()
        self.topic_model_.fit(X,y=y,embeddings=embeddings) # saving fitted labels is not our goal
        self.vocabulary_ = self.topic_model_.vectorizer_model.vocabulary_
        return self.topic_model_.get_topic_info()

    def get_topics(self) -> tuple[pd.DataFrame,pd.DataFrame] :
        """
        Extract the top keywords and their scores from `topic_model_.get_topics(full=False)` output,
        Excluding the Outlier Topic (-1 ID).

        RETURN : 
            (pd.DataFrame,pd.DataFrame) -- each DataFrame shape (n_topics,top_n_words)
                (
                    DataFrame of top characteristic keywords per topic,
                    DataFrame of scores
                )
                
        """
        topics = self.topic_model_.get_topics(full=False)
        words,scores = {},{}
        for k,v in topics.items() : 
            if k==-1 : continue
            words[k],scores[k] = zip(*(v[:self.top_n_words]))
        cols = [f'Top_{_+1}' for _ in range(self.top_n_words)]
        Top_Keywords = pd.DataFrame.from_dict(words,orient='index',dtype='string[pyarrow]',columns=cols).sort_index()
        Top_Scores = pd.DataFrame.from_dict(scores,orient='index',dtype='Float64',columns=cols).sort_index()
        return Top_Keywords,Top_Scores    

    def texts_for_eval(self,documents:list[str]) -> list[list[str]]:
        """
        Tokenize source documents for downstream evaluation (e.g. Gensim coherence),
        using the same preprocessing pipeline and CountVectorizer analyzer as the underlying BERTopic configuration.

        Ref : https://github.com/MaartenGr/BERTopic/issues/90#issuecomment-820915389

        [
            We deliberately avoid concatenating individual documents into topic-level documents.
            Otherwise, a sliding window may proceed through document boundaries and artificially inflate the co-occurrence count.

                topic-level document =
                [doc1.word1, doc1.word2, ..., doc1.wordN,
                doc2.word1, doc2.word2, ..., doc2.wordM]

            For example, a sliding window may incorrectly consider (doc1.wordN,doc2.word1) as neighboring words, 
            even though they originate from different documents.
            We believe such cross-document co-occurrences do not reflect genuine topic coherence.
        ]
        """
        model = self._load_model()
        cleaned_documents = model._preprocess_text(documents)
        # BERTopic's _preprocess_text(...) does not remove punctuation when a custom embedding_model is used
        vectorizer = model.vectorizer_model # Reuse the identical CountVectorizer analyzer used during BERTopic training
        min_len,max_len = vectorizer.ngram_range
        # by default CountVectorizer(...) returns tokens in a following format for ngram_range=(1,2) :
        #       [
        #           'w1', 'w2', ..., 'wN',   <---- starts with all unigrams
        #           'w1 w2', 'w2 w3', ...   <---- followed by all bigrams
        #       ]
        # we instead create an interleaving pattern using nltk :
        #       [ 'w1', 'w1 w2', 'w2', 'w2 w3', ...]
        # the vocabulary remains the same 
        # but the ordering helps us accesing co-occurance of unigram & bigrams in a sliding window
        vectorizer.set_params(ngram_range=(1,1))
        analyzer = vectorizer.build_analyzer()
        return [
            [
                ' '.join(words)
                for words in everygrams(
                    analyzer(doc),
                    min_len=min_len,max_len=max_len,
                )
            ]
            for doc in cleaned_documents
        ]
  



# -------------------------------------------------------------------
class Top_n_BERTopic(TopicModelBase) :
    """
    A convenient wrapper around BERTopic Topic Modeling.
    
    NOTE: This is not a subclass of `TransformerMixin()` API;

        INPUT : 
            top_n_words -- int ; default 10
                The number of words per class to extract.
            embedding_model -- Default 'BAAI/bge-base-en-v1.5'
                The embedding_model that will be used to create embedding vector for keywords or token sets.
                For documents the pre-computed embeddings will be provided at `fit_transform(...)` call.
            n_neighbors -- int ; default 25
                n_neighbors parameter to `umap.UMAP(...)`.
            n_components -- int ; default 5
                n_components parameter to `umap.UMAP(...)`.
            target_weight -- float ; default 0.25
                target_weight parameter to `umap.UMAP(...)`.
            min_cluster_size -- int ; default 100
                min_cluster_size parameter to `hdbscan.HDBSCAN(...)`.
            min_samples -- int ; default 25
                min_samples parameter to `hdbscan.HDBSCAN(...)`.
            gen_min_span_tree -- bool ; default False
                gen_min_span_tree parameter to `hdbscan.HDBSCAN(...)`.
            cluster_selection_method -- Literal['eom','leaf'] ; default 'eom'
                cluster_selection_method parameter to `hdbscan.HDBSCAN(...)`.
            cluster_selection_epsilon -- float ; default 0.0
                cluster_selection_epsilon parameter to `hdbscan.HDBSCAN(...)`.
            stop_words -- list ; default None
                stop_words parameter to `sklearn.feature_extraction.text.CountVectorizer(...)`.
            ngram_range -- tuple ; default (1,2)
                ngram_range parameter to `sklearn.feature_extraction.text.CountVectorizer(...)`.
            diversity -- float ; default 0.2
                diversity parameter to `bertopic.representation.MaximalMarginalRelevance(...)`.
            verbose -- bool ; default True
                verbose parameter to `bertopic.BERTopic(...)`. 
            calculate_probabilities -- bool ; default False
                calculate_probabilities parameter to `bertopic.BERTopic(...)`.
            random_state -- int ; default 42
                random_state for all internal components.
                
    Note: We have kept the Embedding layer outside this wrapper intentionally.
    It helps in scenarios like - 
        1) Compare BERTopic on Unsupervised vs Semi-supervised Modeling
        2) Stability analysis on multiple subset of same data
    Instead of redundant calculations, perform Embedding once and reuse in multiple `fit_transform()` pass.

    [ 
        Several implementation-specific hyperparameters are intentionally hidden to prevent unintended modifications. 
        Refer to the source code for advanced customization. 
    ]

    Ref : https://maartengr.github.io/BERTopic/getting_started/quickstart/quickstart.html, https://maartengr.github.io/BERTopic/getting_started/semisupervised/semisupervised.html
    """
    def __init__(self, top_n_words:int = 10, *, embedding_model = 'BAAI/bge-base-en-v1.5',
            n_neighbors:int = 25, n_components:int = 5, target_weight:float = 0.25, 
            min_cluster_size:int = 100, min_samples:int = 25, gen_min_span_tree:bool = False, 
            cluster_selection_method:Literal['eom','leaf'] = 'eom', cluster_selection_epsilon:float = 0.0,
            stop_words:list|None = None, ngram_range:tuple = (1,2),
            diversity:float = 0.2,
            verbose:bool = True, calculate_probabilities:bool = False,
            random_state:int|None = 42,
        ) :
        super().__init__(top_n_words)
        self.random_state = random_state
        self.embedding_model = embedding_model
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.target_weight = target_weight
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.gen_min_span_tree = gen_min_span_tree
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.diversity = diversity
        self.verbose = verbose
        self.calculate_probabilities = calculate_probabilities

    def fit_transform(self,X:pd.Series,y:pd.Series|None=None,*,embeddings:np.ndarray) -> pd.DataFrame :
        """ 
        INPUT : 
            X -- pandas.Series[str]
                Narratives (Expects clean data from spaCy nlp) 
            y -- pd.Series 
                Class labels
                    y None : Unsupervised Topic Modeling (default)
                    y not None : Semi-supervised Topic Modeling
            embeddings -- numpy.ndarray
                Pre-computed document embeddings

        RETURN : 
            pd.DataFrame -- `get_topic_info()` output of the fitted model

        ATTRIBUTES :
            topic_model_ -- underlying fitted `BERTopic(...)` instance (with all its attributes & methods)
            vocabulary_ -- the underlying vocabulary

        """   
        return super().fit_transform(X,y,embeddings=embeddings)
    
    def _load_model(self) -> BERTopic :
        # choose n_jobs=1, so we can run a outer parallel loop of multiple fits
        """Create unfitted BERTopic instance."""
        _UMAP = UMAP(# [UMAP with optimized initialization for large data]
            n_neighbors=self.n_neighbors,min_dist=0.0, 
            n_components=self.n_components,
            metric='cosine', # sentence embeddings are designed for cosine similarity
            random_state=self.random_state,transform_seed=self.random_state,n_jobs=1, # mandatory for reproducability
            low_memory=True,
            unique=True, # duplicate points remain duplicate in output, creating high-density region
            init='tswspectral',
            target_weight=self.target_weight, # balance the weightage between data and label
            densmap=False,
        )
        _HDBSCAN = HDBSCAN(# [HDBSCAN with prediction enabled]
            min_cluster_size=self.min_cluster_size,min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric='euclidean', # UMAP already tightly packed similar points together
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True, 
            gen_min_span_tree=self.gen_min_span_tree,
            allow_single_cluster=False,
            core_dist_n_jobs=1, # HDBSCAN getting finished in 10-15sec, parallel overhead not required
        )
        _COUNTVECTORIZER = CountVectorizer(# [CountVectorizer with stop_words removal]
            lowercase=True,analyzer='word',
            tokenizer=str.split,token_pattern=None, # expecting clean data from spacy
            stop_words=list(map(str.lower,self.stop_words)) if self.stop_words is not None else None,
            ngram_range=self.ngram_range,
            min_df=1,max_df=0.95,max_features=None,
            # CountVectorizer is fitted on topic-level documents ; therefore do not filter min_df else you may lose unique words for a topic
            # max_df=0.95 removes words shared by more than 95% of the discovered topics
            # when the topics are imbalanced, putting any filter on max_features will tend to remove tokens from minority topics due to smaller absolute count
        ) 
        _CTFIDF = ClassTfidfTransformer(# [ClassTfidfTransformer with reduce_frequent_words]
            bm25_weighting=True,reduce_frequent_words=True,
            # these configurations heavily penalize the generic words
        )
        _REPRESENTATION = [# [KeyBERT+MMR]
            KeyBERTInspired(
                top_n_words=4*self.top_n_words,nr_candidate_words=10*self.top_n_words,
                nr_repr_docs=20,
                random_state=self.random_state,
            ), # prioritize keywords that are semantically aligned with the underlying topic
            MaximalMarginalRelevance(diversity=self.diversity,top_n_words=self.top_n_words) # penalize repeated keywords
        ]
        return BERTopic(
            verbose=self.verbose,
            top_n_words=self.top_n_words,n_gram_range=self.ngram_range,
            embedding_model=self.embedding_model, 
            umap_model=_UMAP,
            hdbscan_model=_HDBSCAN,
            vectorizer_model=_COUNTVECTORIZER, 
            ctfidf_model=_CTFIDF,
            representation_model=_REPRESENTATION,
            calculate_probabilities=self.calculate_probabilities,
        )

# -------------------------------------------------------------------




# -------------------------------------------------------------------
class Top_n_cTFIDF(TopicModelBase) :
    """
    A convenient wrapper around BERTopic Manual Topic Modeling.

    NOTE: This is not a subclass of `TransformerMixin()` API;

        INPUT : 
            top_n_words -- int ; default 10
                The number of words per class to extract.
            ngram_range -- tuple ; default (1,2)
                ngram_range parameter to `sklearn.feature_extraction.text.CountVectorizer(...)`.
            stop_words -- list ; default None
                stop_words parameter to `sklearn.feature_extraction.text.CountVectorizer(...)`.
    [ 
        Several implementation-specific hyperparameters are intentionally hidden to prevent unintended modifications. 
        Refer to the source code for advanced customization. 
    ]

    Ref : https://maartengr.github.io/BERTopic/getting_started/manual/manual.html
    """
    def __init__(self,top_n_words=10,*,ngram_range=(1,2),stop_words:list=None) :
        super().__init__(top_n_words)
        self.ngram_range = ngram_range
        self.stop_words = stop_words

    def fit_transform(self,X:pd.Series,y:pd.Series) -> pd.DataFrame :
        """ 
        INPUT : 
            X -- pandas.Series[str]
                Narratives (Expects clean data from spaCy nlp) 
            y -- pandas.Series
                Class labels 

        RETURN : 
            pd.DataFrame -- `get_topic_info()` output of the fitted model

        ATTRIBUTES :
            topic_model_ -- underlying fitted `BERTopic(...)` instance (with all its attributes & methods)
            vocabulary_ -- the underlying vocabulary
            Narrative_Classes_ -- Index containing class labels ; shape (n_classes,)
            cTfIdf_Matrix_ -- SciPy.csr_matrix of scores ; shape (n_classes,len(Token_Names_))

        """ 
        y = pd.Categorical(y) 
        self.Narrative_Classes_ = y.categories
        out = super().fit_transform(X,y.codes)
        self.cTfIdf_Matrix_ = self.topic_model_.c_tf_idf_
        return out

    def _load_model(self) -> BERTopic :
        """Create unfitted BERTopic instance."""
        return BERTopic(
            top_n_words = self.top_n_words,n_gram_range=self.ngram_range,
            embedding_model=BaseEmbedder(), # No embedding
            umap_model=BaseDimensionalityReduction(), # No dimensionality reduction
            hdbscan_model=BaseCluster(), # No clustering
            vectorizer_model=CountVectorizer(
                lowercase=True,stop_words=self.stop_words,analyzer='word', 
                tokenizer=str.split,token_pattern=None, # expecting clean data from spacy
                ngram_range=self.ngram_range,
                min_df=1,max_df=0.9,max_features=None,
                # CountVectorizer is fitted on class-level documents (~10); therefore do not filter min_df else you may lose unique words for a topic
                # max_df=0.9 removes words shared by more than 90% of the discovered topics
                # when the topics are imbalanced, putting any filter on max_features will tend to remove tokens from minority topics due to smaller absolute count
            ), 
            ctfidf_model=ClassTfidfTransformer(
                bm25_weighting=True,reduce_frequent_words=True,
                # these configurations heavily penalize the generic words
            ),
            representation_model=None,
            calculate_probabilities=False,
        )

    def get_topics(self) -> tuple[pd.DataFrame,pd.DataFrame] :
        """
        Extract the top keywords and their scores.

        RETURN : 
            (pd.DataFrame,pd.DataFrame) -- each DataFrame shape (n_classes,top_n_words)
                (
                    DataFrame of top characteristic keywords per topic,
                    DataFrame of scores
                )
                
        """
        Top_Keywords,Top_Scores = super().get_topics()
        Top_Keywords.index = self.Narrative_Classes_
        Top_Scores.index = self.Narrative_Classes_
        return Top_Keywords,Top_Scores    
    
# -------------------------------------------------------------------





