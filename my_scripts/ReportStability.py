from sklearn.base import BaseEstimator
from collections.abc import Sequence,Callable
import numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.metrics import adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from itertools import chain



class TopicQuality(BaseEstimator) :
    """
    A wrapper function,
    that computes some commonly used topic coherence/ topic diversity/ topic stability measures
    out of our simulation data.

    [Internal Use Only]

    """
    def __init__(self,top_n_words:int=10) :
        self.top_n_words = top_n_words

    def _info(self,counts_all:Sequence[pd.Series],) -> list[pd.Series] :
        noise = 100*pd.Series([x.get(-1,0)/x.sum() for x in counts_all],dtype='Float64',name='noise rate')
        n_topic = pd.Series([len(x.drop(-1,errors='ignore')) for x in counts_all],dtype='Int32',name='topics')
        return [n_topic,noise,]

    def _size(self,counts_all:Sequence[pd.Series],) -> list[pd.Series] :
        out_size = pd.Series(index=range(len(counts_all)),dtype='string',name='topic size (min,max,median)')
        for itr,count_topics in enumerate(counts_all) :
            size = count_topics.drop(-1,errors='ignore').astype('Int32')
            out_size[itr] = f"{size.min()} ; {size.max()} ; {size.median():.1f}"
        return [out_size,]

    def _topic_coherence(self,topics_all:Sequence[list[list[str]]],texts_all:Sequence[list[list[str]]],topic_word_embeddings_all:Sequence[list[np.ndarray]],) -> list[pd.Series] :
        # EVALUATE: NPMI coherence ---
        # NPMI is often stated as showing great alignment with human judgements
        out_npmi = pd.Series(index=range(len(topics_all)),dtype='Float64',name='NPMI coherence')
        for itr,(topics,texts) in enumerate(zip(topics_all,texts_all)) :
            out_npmi.iloc[itr] = np.median(
                CoherenceModel(
                    topics=topics,texts=texts,dictionary=Dictionary(texts),
                    coherence='c_npmi',
                    window_size=10*2, 
                    # we are using interleaving unigrams and bigrams : 
                    #   [w1, w1 w2, w2, w2 w3, ...]
                    # n-words of original sentence consumes 2n-1 tokens
                    topn=self.top_n_words,processes=8
                ).get_coherence_per_topic() 
            )
        # ---
        # EVALUATE: WETC coherence ---
        # since KeyBERTInspired selects the top words based on embedding similarity
        out_wetc = pd.Series(index=range(len(topic_word_embeddings_all)),dtype='Float64',name='WETC_pw coherence')
        for itr,topic_word_embeddings in enumerate(topic_word_embeddings_all) :
            out_wetc.iloc[itr] = np.median(
                [
                    self._pairwise_avg(words_embedding,cosine_similarity) 
                    for words_embedding in topic_word_embeddings
                ]
            )
        # ---
        return [out_npmi,out_wetc,]
   
    def _topic_diversity(self,topics_all:Sequence[list[list[str]]],topic_embeddings_all:Sequence[np.ndarray],) -> list[pd.Series] :
        # EVALUATE: TD diversity ---
        # the most straight-forward diversity measure to report
        out_td = pd.Series(index=range(len(topics_all)),dtype='Float64',name='TD diversity')
        for itr,topics in enumerate(topics_all) :
            words = set(chain.from_iterable(topics))
            out_td.iloc[itr] = len(words)/(len(topics)*self.top_n_words)
        # ---
        # EVALUATE: Centroid_Distance diversity on topic_embeddings_ 
        # BERTopic uses topic_embeddings_ in downstream methods such as find_topics(), merge_models(), and approximate_distribution() 
        out_cd = pd.Series(index=range(len(topic_embeddings_all)),dtype='Float64',name='Centroid_Distance diversity')
        for itr,topic_embeddings in enumerate(topic_embeddings_all) :
            out_cd.iloc[itr] = self._pairwise_avg(topic_embeddings,cosine_distances,agg=np.median)
        # ---
        return [out_td,out_cd,]
    
    def _topic_stability(self,cross_subset_similarity_all:Sequence[np.ndarray],) -> list[pd.Series] :
        # EVALUATE: best matched pairwise cosine similarity on topic_embeddings_
        # the best matching topic pairs between two runs of the model are found via solving a linear assignment problem
        out_stability = pd.Series(index=range(len(cross_subset_similarity_all)),dtype='Float64',name='cross-subset topic stability')
        for itr,cross_subset_similarity in enumerate(cross_subset_similarity_all) :
            r_id,c_id = linear_sum_assignment(cross_subset_similarity,maximize=True)
            out_stability.iloc[itr] = np.median(cross_subset_similarity[r_id,c_id])
        # ---
        return [out_stability,] 
    
    def _cluster_stability(self,fit_vs_predicted_clustering_all:Sequence[tuple[list,list]],) -> list[pd.Series] :
        # EVALUATE: AMI stability ---
        # we prefer AMI over ARI as the clusters are highly imbalanced in sizes
        # we report cluster stability along with topic stability, as per the modern practice where 
        # we can entirely skip the topic words generation step, and instead produce a LLM-written summary out of the cluster representatives
        out_ami = pd.Series(index=range(len(fit_vs_predicted_clustering_all)),dtype='Float64',name='AMI cluster stability')
        for itr,(labels_true,labels_pred) in enumerate(fit_vs_predicted_clustering_all) :
            out_ami.iloc[itr] = adjusted_mutual_info_score(labels_true,labels_pred)
        # ---
        return [out_ami,]       


    def evaluate_all(self,
            counts_all:Sequence[pd.Series],
            topics_all:Sequence[list[list[str]]],
            topic_word_embeddings_all:Sequence[list[np.ndarray]],
            topic_embeddings_all:Sequence[np.ndarray],
            fit_vs_predicted_clustering_all:Sequence[tuple[list[int],list[int]]],
            cross_subset_similarity_all:Sequence[np.ndarray],
            texts_all:Sequence[list[list[str]]],
        ) -> pd.DataFrame :
        """ Evaluate multiple runs : compute within sample & between sample performance metrices. """
        topics_all = [
            [topic[:self.top_n_words] for topic in topics]
            for topics in topics_all
        ]
        topic_word_embeddings_all = [
            [words_embedding[:self.top_n_words] for words_embedding in topic_word_embeddings]
            for topic_word_embeddings in 
            topic_word_embeddings_all
        ]
        df_2r = pd.concat(# for r-pairs of subsamples in experiment, this df will have 2r rows
            [
                *(self._info(counts_all,)),
                *(self._topic_coherence(topics_all,texts_all,topic_word_embeddings_all,)),
                *(self._topic_diversity(topics_all,topic_embeddings_all,)),
                *(self._cluster_stability(fit_vs_predicted_clustering_all,)),
            ], axis=1
        )
        df_r = pd.concat(# for r-pairs of subsamples in experiment, this df will have r rows
            [
                *(self._topic_stability(cross_subset_similarity_all,)),
            ], axis=1
        )
        df_2r.index = pd.MultiIndex.from_product([[str(_+1) for _ in range(len(df_r))],["A","B"]],names=["pair","subset"])
        df_r.index = pd.Index([str(_+1) for _ in range(len(df_r))],name="pair")
        # Pair-level metrics are duplicated for subsets A and B
        # duplicating each number of a list exactly twice does not change its empirical distribution
        # allowing mean, variance, quantiles to be calculated from the combined table
        out = df_2r.join(df_r,on="pair")
        return out
    
    def evaluate(self,
            counts:pd.Series,
            topics:list[list[str]],
            topic_word_embeddings:list[np.ndarray],
            topic_embeddings:np.ndarray,
            texts:list[list[str]],
            index_label=0
        ) -> pd.DataFrame :
        """ Evaluate single run : compute within sample performance metrices. """
        topics = [topic[:self.top_n_words] for topic in topics]
        topic_word_embeddings = [words_embedding[:self.top_n_words] for words_embedding in topic_word_embeddings]
        out = pd.concat(# df with single row
            [
                *(self._info([counts],)),
                *(self._size([counts],)),
                *(self._topic_coherence([topics],[texts],[topic_word_embeddings],)),
                *(self._topic_diversity([topics],[topic_embeddings],)),
            ], axis=1
        )
        out.index = [index_label]
        return out


    @staticmethod
    def _pairwise_avg(x:np.ndarray,func_pairwise:Callable,*,agg:Callable=np.mean) -> float :
        # Average upper-triangular pairwise score
        # (excluding diagonal)
        assert len(x)>=2, "pairwise comparison undefined for len(array)<2"
        out = func_pairwise(x)
        return agg(out[np.triu_indices_from(out,k=1)])


