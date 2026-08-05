""" 
    It helps us extracting topic centroid embeddings and topic words embeddings from a fitted topic model,
    well-organised in a dataclass format.
"""
from dataclasses import dataclass
import numpy as np, pandas as pd



@dataclass(frozen=True)
class Topics :
    topics: list[list[str]]
    topic_embeddings: np.ndarray
    topic_word_embeddings: list[np.ndarray]|None 

def get_embeddings_for_topics(topic_model,Representation:pd.Series,word_embeddings:bool=True) -> Topics :
    """
    From a fitted topic model extract topic embeddings (excluding outlier class).

    [Internal Use Only]
    
    INPUT :    
        topic_model -- fitted BERTopic(...) object.
        Representation -- pd.Series
            "Representation" of get_topic_info() output, with "Topic" as index.
        word_embeddings -- bool ; default True
            If True, embed topic words per topic.

    RETURN :
        Topics(...) -- dataclass instance.

    """
    exclude_outlier = (Representation.index != -1) 
    topics = [
        list(filter(None,words))
        # sometimes, the model fails to find enough representative words and returns '' as padding
        # we remove the padding while keeping the remaining words as they are
        for words in (
            Representation.loc[exclude_outlier]
        )
    ]
    topic_embeddings = topic_model.topic_embeddings_[exclude_outlier]
    if word_embeddings :
        emb = topic_model.embedding_model
        topic_word_embeddings = [emb.embed_words(topic) for topic in topics]
    else : topic_word_embeddings = None
    return Topics(topics,topic_embeddings,topic_word_embeddings)