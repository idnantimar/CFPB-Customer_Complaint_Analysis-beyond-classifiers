import numpy as np



def _Rand_OddEven(X:np.ndarray,seed:int) :
    n = len(X)
    n_even = (n+1)//2 ; n_odd = n-n_even
    AB_mask = np.empty(n,dtype=bool)  # True: Group-A , False: Group-B
    rng = np.random.default_rng(seed) 
    AB_mask[0::2] = rng.integers(0,2,size=n_even,dtype=bool)
    AB_mask[1::2] = ~AB_mask[0:(2*n_odd):2]
    return (X[AB_mask],X[~AB_mask])

def split_OddEvenRand(*X,stratify=None,seed=None) -> tuple :
    """ 
    Splits data into two equal halves for half-split stability analysis.
    Maintains strict chronological timeline ordering while enforcing stratification.

    First creates pair like (0,1)(2,3)(4,5),...
    Within each pair, randomly allocate Group-A/Group-B

    Note: 
    [1] Unlike `sklearn.model_selection.train_test_split(...)` -
        it allows stratify (handle imbalanced data) 
        but does not perform shuffle (preserve chronological order).
    [2] Unlike `sklearn.model_selection.StratifiedKFold(...)` or `sklearn.model_selection.TimeSeriesSplit(...)` - 
        it splits the data in odd-even basis, so both segment has equal representation of entire timeline. 

    INPUT : 
        X -- sequence of pandas.DataFrame/pandas.Series
            Each item is assumed (not validated) to have same index    
        stratify -- pd.Series with same index as X ; default None
        seed -- seed for `numpy.random.default_rng(...)` 

    RETURN : 
        tuple -- len is 2*len(X) ; similar format as sklearn.model_selection.train_test_split()
            
    INPLACE MODIFY :
        <NA>

    """
    n = len(X[0])
    if any(len(x) != n for x in X[1:]): raise ValueError("All inputs X must have the same length.")
    if stratify is not None :
        if len(stratify) != n : raise ValueError("stratify must have the same length as X.")
        idx_full = np.argsort(stratify.to_numpy(),kind='stable') 
        # similar class labels will become contiguous if we rerrange data by idx_full
        # original order is preserved under each labels by 'stable'
        # rows reshuffled, but still holds the original row positions that can be passed to .iloc[]
    else : idx_full = np.arange(n) # default 0,1,2,...
    idx_A,idx_B = _Rand_OddEven(idx_full,seed)
    idx_A,idx_B = np.sort(idx_A),np.sort(idx_B) # restore original observed row orders within each group 
    out = []
    for x in X : out.extend((x.iloc[idx_A],x.iloc[idx_B]))

    return tuple(out)