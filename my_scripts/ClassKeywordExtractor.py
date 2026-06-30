import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.cluster import BaseCluster
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction



embedding_model_NULL = BaseEmbedder()
dimensionality_model_NULL = BaseDimensionalityReduction()
cluster_model_NULL = BaseCluster()
def vectorizer_model_ngram(ngram_range:tuple=(1,2),*,max_df:float=0.8,stop_words=None) :
    return CountVectorizer(
        lowercase=True, stop_words=stop_words, analyzer='word', 
        ngram_range=ngram_range,
        min_df=1, max_df=max_df, max_features=None
        # CountVectorizer is applied on class level doc, the total number is limited (~10) ; don't filter min_df
        # when the classes are imbalanced, putting any filter on max_features will tend to remove tokens from minority classes due to smaller absolute count
    )
ctfidf_model_REGULARIZED = ClassTfidfTransformer(
    bm25_weighting=True,reduce_frequent_words=True
    # these configurations heavily penalize the generic words
)



class Top_n_cTFIDF(BaseEstimator) :
    """
    A convenient wrapper around BERTopic() Manual Topic Modeling.

    Note: This cannot be a subclass of `TransformerMixin()` API; because transform(X_test) fails at step-2 without y_test.

        INPUT : 
            top_n_words -- int ; default 10
                The number of words per class to extract.
            ngram_range -- tuple ; default (1,2)
                The lower and upper boundary of the range of n-values for different word n-grams.

    [ 
        Several implementation-specific hyperparameters are intentionally hidden to prevent unintended modifications. 
        Refer to the source code for advanced customization. 
    ]

    Ref : https://maartengr.github.io/BERTopic/getting_started/manual/manual.html
    """
    def __init__(self,*,top_n_words=10,ngram_range=(1,2)) :
        self.top_n_words = top_n_words
        self.ngram_range = ngram_range

    def fit_transform(self,X:pd.Series,y:pd.Series) -> pd.DataFrame :
        """ 
        INPUT : 
            X -- pandas.Series[str]
                Narratives (Expects clean data from spaCy) 
            y -- pandas.Series
                Class labels 

        RETURN : 
            pd.DataFrame -- shape (n_classes,top_k)
                Each row corresponds to a class label of y,
                For that row the top_k keywords are arranged in a decreasing order of scores. 
                
        INPLACE MODIFY :
            <NA>

        ATTRIBUTES :
            topic_model_ -- underlying fitted BERTopic() instance
            cTfIdf_Matrix_ -- SciPy.csr_matrix of scores ; shape (n_classes,len(Token_Names_))
            Narrative_Classes_ -- Index containing class labels ; shape (n_classes,)
            Token_Names_ -- the underlying vocabulary
            Top_Keywords_ -- pd.DataFrame of top characteristic keywords per class ; shape (n_classes,top_k)
            Top_Scores_ -- pd.DataFrame of scores corresponding to Top_Keywords_ ; shape (n_classes,top_k)

        """   
        ## ------- fit ------------
        X,y = X.to_list(),pd.Categorical(y)
        self.Narrative_Classes_ = y.categories
        self.topic_model_ = BERTopic(
            top_n_words = self.top_n_words,
            embedding_model=embedding_model_NULL, # No embedding
            umap_model=dimensionality_model_NULL, # No dimensionality reduction
            hdbscan_model=cluster_model_NULL, # No clustering
            vectorizer_model=CountVectorizer(
                lowercase=True,stop_words=None,analyzer='word', 
                tokenizer=str.split, # expecting clean data from spacy
                ngram_range=self.ngram_range,
                min_df=1, max_df=0.6, max_features=None
                # CountVectorizer is applied on class level doc, the total number is limited (~10) ; don't filter min_df
                # when the classes are imbalanced, putting any filter on max_features will tend to remove tokens from minority classes due to smaller absolute count
                # any word common in >60% classes is removed ; remove shared words across classes early
            ), 
            ctfidf_model=ctfidf_model_REGULARIZED,
            representation_model=None,
            calculate_probabilities=False
        )
        _ = self.topic_model_.fit_transform(X,y=y.codes) # we already know which doc belongs to which class
        self.Token_Names_ = self.topic_model_.vectorizer_model.get_feature_names_out()
        self.cTfIdf_Matrix_ = self.topic_model_.c_tf_idf_
        ## -------
        ## ------- score ------------
        topics = self.topic_model_.get_topics()
        words = {
            self.Narrative_Classes_[k]: [w for w,_ in v]
            for k,v in topics.items()
        }
        scores = {
            self.Narrative_Classes_[k]: [s for _,s in v]
            for k,v in topics.items()
        }
        cols = [f'Top_{_}' for _ in range(1,self.top_n_words+1)]
        self.Top_Keywords_ = pd.DataFrame.from_dict(words,orient='index',dtype='string',columns=cols)
        self.Top_Scores_ = pd.DataFrame.from_dict(scores,orient='index',dtype='Float64',columns=cols)
        ## ------- 
        return self.Top_Keywords_




