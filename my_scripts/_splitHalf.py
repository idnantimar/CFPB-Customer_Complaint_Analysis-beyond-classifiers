import numpy as np, pandas as pd


def _OddEven(X:np.ndarray) :
    n = len(X)
    AB_mask = np.array([True,False]*((n+1)//2))[:n] # True: Group-A , False: Group-B
    return (X[AB_mask],X[~AB_mask])


def split_OddEven(*X,stratify=None) :
    """ 
    Splits data into two equal halves for half-split stability analysis.
    Maintains strict chronological timeline ordering while enforcing stratification. 

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

    RETURN : 
        tuple -- len is 2*len(X) ; similar format as sklearn.model_selection.train_test_split()
            
    INPLACE MODIFY :
        <NA>

    """
    if stratify is not None :
        idx_full = np.argsort(stratify,kind='stable') 
        # similar class labels will become contiguous if we rerrange data by idx_full
        # original order is preserved under each labels by 'stable'
        # rows reshuffled, but still holds the original row positions that can be passed to .iloc[]
    else : idx_full = np.arange(len(X[-1])) # default 0,1,2,...
    idx_A,idx_B = _OddEven(idx_full)
    idx_A,idx_B = np.sort(idx_A),np.sort(idx_B) # restore original observed row orders within each group 
    out = ()
    for x in X : out += (x.iloc[idx_A],x.iloc[idx_B])

    return out

