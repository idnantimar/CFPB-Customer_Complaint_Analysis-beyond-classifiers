import numpy as np, pandas as pd
from collections.abc import Iterable
import spacy ; spacy.require_cpu()



nlp = spacy.load("en_core_web_sm",disable=["parser","ner"])
def spaCy_cleaner(S:pd.Series,*,stop_words:Iterable=None,exclude_pos:Iterable=None,exclude_tag=None,lemma=True,**nlp_kwargs) -> pd.Series:
    """
    The default regex based tokenization in sklearn has some pitfalls: 
        Numbers like 1,23,456.78 are tokenized into four separate tokens separated by commas and decimal points.
        It does not consider lemmatization, e.g.- words like "dispute"/"disputed"/"disputing" are treated as different tokens bloating the vocabulary.
    To mitigate these issues, we use spaCy pretrained NLP models here to tokenize the narratives before feeding them into sklearn API.
    downstream calculations remain the same as the default, but over a better vocabulary. 

    INPUT :
        S -- pandas.Series
            Narrative
        stop_words -- list ; default None
            The list of generic words (case-insensitive), all of which must be removed from analysis.
        exclude_pos -- list ; default None
            The list of token.pos_ to be removed from vocabulary.
            exclude_pos=None removes {'X','SYM','SPACE','EOL'}
        exclude_tag -- list ; default None
            The list of token.tag_ to be removed from vocabulary.
            exclude_tag=None removes {}
        lemma -- bool ; default True
            If True, replace token.text_ with token.lemma_ 
        nlp_kwargs -- kwargs to nlp.pipe(...) ; e.g. n_process=8, batch_size=1000

    [
        Use of generic stop_words list is usually not encouraged, unless curated properly.
        Here we try to filter out few token.pos_ & token.tag_ (https://spacy.io/api/token) that carries little to no extra information in n-gram keywords.    
    ]
            
    RETURN :
        pd.Series
            Cleaned data where tokens are separated by ' ' ; shape same as S.shape 

    INPLACE MODIFY :
        <NA>
    
    """
    docs = nlp.pipe(S.fillna("").to_list(),**nlp_kwargs)
    stop_words_ = {str(_).strip().lower() for _ in stop_words} if stop_words is not None else set()
    exclude_pos = set(exclude_pos) if exclude_pos is not None else {'X','SYM','SPACE','EOL'}
    exclude_tag = set(exclude_tag) if exclude_tag is not None else {}

    return pd.Series(
        [
                " ".join(
                    token.lemma_ if lemma else token.text
                    for token in doc 
                    if not (
                        token.is_punct or token.is_space
                        or token.pos_ in exclude_pos or token.tag_ in exclude_tag
                        or token.text.lower() in stop_words_
                    )
                ) for doc in docs 
        ], 
        # don't lowercase unnecessarily, until required at specific stages.
        # spaCy's lemmatizer already lowercases standard nouns or verbs or adjectives and retain original casing of proper nouns; this can be leveraged in some new-age models.
        index=S.index, dtype='string'
    )