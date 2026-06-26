import pandas as pd
from sklearn.base import BaseEstimator
import spacy ; spacy.require_cpu()
from spacy.parts_of_speech import IDS
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer


nlp = spacy.load("en_core_web_sm",disable=["parser","ner"])
POS_ALL = IDS.keys()-{''} # set of all parts-of-speech labels
def _spaCy_tokenizer(S:pd.Series,*,stop_words:set=None,exclude_pos:set=None) :
    # The default regex based tokenization in sklearn has some pitfalls: 
        # Numbers like 1,23,456.78 are tokenized into four separate tokens separated by commas and decimal points.
        # It does not consider lemmatization, e.g.- words like "dispute"/"disputed"/"disputing" are treated as different tokens bloating the vocabulary.
    # To mitigate these issues, we use spaCy pretrained NLP models here to tokenize the narratives before feeding them into sklearn API.
    # downstream calculations remain the same as the default, but over a better vocabulary.  
    docs = nlp.pipe(S.to_list(),n_process=8,batch_size=1000)
    stop_words = {str(_).lower() for _ in stop_words} if stop_words is not None else set()
    exclude_pos = set(exclude_pos) if exclude_pos is not None else POS_ALL-{'NOUN','PROPN','ADJ','VERB','NUM'}
    # use of generic stop_words list is usually not encouraged, unless curated properly
    # here we try to filter out few token.pos_(https://spacy.io/api/token),token.tag_(https://spacy.io/models/en#en_core_web_sm) that carries little to no extra information in n-gram keywords 
    # underlying rationale : 
        # the gist of a complaint is hidden inside Noun/Adjective/Verb ; signal-to-noise ratio for other tokens is very very less 
        # additionally in financial market context, a specific amount may surface in repeated complaints that we donot want to lose
    # while okay for keyword extraction, be cautious in removing words like 'not'(PART)/'without'(ADP)/'neither'(CCONJ) if end goal is sentiment analysis
    return pd.Series(
        [
                " ".join(
                    token.lemma_
                    for token in doc 
                    if not (
                        token.is_punct or token.is_space
                        or token.pos_ in exclude_pos
                        or token.text.lower() in stop_words
                    )
                ) for doc in docs 
        ], 
        # don't lowercase unnecessarily, until required at specific stages.
        # spaCy's lemmatizer already lowercases standard nouns or verbs or adjectives and retain original casing of proper nouns; this can be leveraged in some new-age models.
        index=S.index, dtype='string'
    )


class TopK_cTFIDF(BaseEstimator) :
    """
    The available ClassTfidfTransformer() in `bertopic` does not accept raw documents, but a precomputed token counts matrix over class level macro-documents.
    Instead of doing the steps one by one, this is a convenient wrapper around the CFPB cTFIDF keyword extraction pipeline -
        1) Tokenize raw documents,
        2) Construct class level macro-documents,
        3) Compute class level token counts matrix,
        4) Fit the ClassTfidfTransformer(),
        5) Extract top keywords for each class from a SciPy.csr_matrix.

    Note: This cannot be a subclass of `TransformerMixin()` API; because transform(X_test) fails at step-2 without y_test.

        INPUT : 
            top_k -- int ; default 20
                The number of top keywords to select per class.
            ngram_range -- tuple ; default (1,2)
                The lower and upper boundary of the range of n-values for different word n-grams.
            stop_words -- list ; default None
                The list of generic words (case-insensitive), all of which must be removed from analysis.
                Note : anything `token.pos_ in exclude_pos` or `max_df>0.4` will be dropped by implementation.
            exclude_pos -- set ; default None
                The set of token.pos_ to be removed from vocabulary.
                exclude_pos=None removes {'X','SYM','SPACE','EOL','CCONJ','SCONJ','INTJ','ADP','DET','PART','AUX','ADV','PRON'}
                Rationale : The gist of a financial complaint is primarily captured by nouns, adjectives, verbs, and occasionally numbers.
            warm_start -- bool ; default False
                If True, it ignores X completely and reuse available `Narrative_Tokenized_` during `fit()`. 
                It is useful, if you want to explore multiple y or ngram_range or top_k on same X. 
                example -
                    >>> my_TopK = TopK_cTFIDF(warm_start=True)
                    >>> Top10_perY1 = my_TopK.fit_transform(X,y1)
                    >>> Top10_perY2 = my_TopK.fit_transform(X,y2)
                    >>> my_TopK.set_params(top_k=20,ngram_range=(1,2))
                    >>> Top20_bi_perY1 = my_TopK.fit_transform(X,y1)
                warm_start avoids the heavy nlp() stage over and over at step-1. 
                WARNING : 
                    When warm_start=True, `Narrative_Tokenized_` is reused without verifying that X is identical to the previous session.
                    warm_start does not apply for stop_words & exclude_pos.
    [ 
        Several implementation-specific hyperparameters are intentionally hidden to keep the public API concise. 
        Refer to the source code for advanced customization. 
    ]

    """
    def __init__(self,*,top_k=20,ngram_range=(1,2),stop_words=None,exclude_pos=None,warm_start=False) :
        self.top_k = top_k
        self.ngram_range = ngram_range
        self.stop_words = {str(_).lower() for _ in stop_words} if stop_words is not None else set()
        self.exclude_pos = set(exclude_pos) if exclude_pos is not None else POS_ALL-{'NOUN','PROPN','ADJ','VERB','NUM'}
        self.warm_start = warm_start

    def fit_transform(self,X:pd.Series,y:pd.Series) -> pd.DataFrame :
        """ 
            INPUT : 
                X -- pandas.Series[str]
                    Narratives
                y -- pandas.Series 
                    Class labels 

            RETURN : 
                pd.DataFrame -- shape (n_classes,top_k)
                    Each row corresponds to a class label of y,
                    For that row the top_k keywords are arranged in a decreasing order of scores. 
                
            INPLACE MODIFY :
                <NA>

            ATTRIBUTES :
                cTfIdf_ -- trained ClassTfidfTransformer() instance
                cTfIdf_Matrix_ -- SciPy.csr_matrix of scores ; shape (n_classes,len(Token_Names_))
                Narrative_Classes_ -- pd.Series containing collated tokens per class ; shape (n_classes,)
                Token_Count_Matrix_ -- SciPy.csr_matrix of token absolute counts ; shape (n_classes,len(Token_Names_))
                Token_Names_ -- the underlying vocabulary
                Top_Keywords_ -- pd.DataFrame of top characteristic keywords per class ; shape (n_classes,top_k)
                Top_Scores_ -- pd.DataFrame of scores corresponding to Top_Keywords_ ; shape (n_classes,top_k)

        """   
        ## ---- STEP-1 -------------------------------     
        if hasattr(self,'Narrative_Tokenized_') and self.warm_start : 
            pass
        else : self.Narrative_Tokenized_ = _spaCy_tokenizer(X,stop_words=self.stop_words,exclude_pos=self.exclude_pos)
        ## ----------------------------------- 
        ## ---- STEP-2 ------------------------------- 
        self.Narrative_Classes_ = self.Narrative_Tokenized_.groupby(y,as_index=True,sort=False,dropna=False).agg(" ".join)
        ## ----------------------------------- 
        ## ---- STEP-3 ------------------------------- 
        vectorizer = CountVectorizer(
            # since we have already pre-tokenized the narratives, we can simply split by whitespace here.
            lowercase=True, stop_words=None, analyzer='word', tokenizer=str.split, 
            ngram_range=self.ngram_range,
            min_df=1, max_df=0.4, max_features=None
            # CountVectorizer is applied on class level doc, the total number is limited (~10) ; don't filter min_df
            # when the classes are imbalanced, putting any filter on max_features will tend to remove tokens from minority classes due to smaller absolute count
        )
        self.Token_Count_Matrix_ = vectorizer.fit_transform(self.Narrative_Classes_)
        self.Token_Names_ = vectorizer.get_feature_names_out()
        ## ----------------------------------- 
        ## ---- STEP-4 ------------------------------- 
        self.cTfIdf_ = ClassTfidfTransformer(
            bm25_weighting=True, 
            reduce_frequent_words=True
            # these configurations heavily penalize the generic words
        )
        self.cTfIdf_Matrix_ = self.cTfIdf_.fit_transform(self.Token_Count_Matrix_)
        ## ----------------------------------- 
        ## ---- STEP-5 ------------------------------- 
        Top_Keywords,Top_Scores = {},{}
        # cTfIdf_Matrix_ is a sparse matrix, where top_k is a very small fraction of len(Token_Names_)
        # manipulating SciPy.csr_matrix is far more efficient than explicit conversion to dense form
        # also avoid sorting of a large array (~10K) to get top k elements (~10) ; use partition instead , timecomplexity O(nlogn)-->O(nlogk)
        for class_idx,start_pt,end_pt in zip(self.Narrative_Classes_.index,self.cTfIdf_Matrix_.indptr[:-1],self.cTfIdf_Matrix_.indptr[1:]):
            key,value = self.cTfIdf_Matrix_.indices[start_pt:end_pt],self.cTfIdf_Matrix_.data[start_pt:end_pt]
            k = min(self.top_k,end_pt-start_pt)
            sorted_seq = value.argpartition(kth=-k)[-k:] ; sorted_seq = sorted_seq[value[sorted_seq].argsort()][::-1]
            Top_Keywords[class_idx],Top_Scores[class_idx] = self.Token_Names_[key[sorted_seq]],value[sorted_seq]
        cols=[f'Top_{_}' for _ in range(1,self.top_k+1)]
        self.Top_Keywords_ = pd.DataFrame.from_dict(Top_Keywords,orient='index',dtype='string',columns=cols)
        self.Top_Scores_ = pd.DataFrame.from_dict(Top_Scores,orient='index',dtype='Float64',columns=cols)
        ## ----------------------------------- 
        return self.Top_Keywords_




if __name__=='__main__' : 
    pass
