import pandas as pd
import ast


def clean_text(Source_file) :
    """ 
    Read the raw CFPB data from a given path and return it with cleaned and trimmed data.

    INPUT : 
        Source_file -- file for pandas.read_csv(...)

    RETURN : 
        Data_Narrative -- pandas.DataFrame 
            "DateReceived":'period[D]' | "Product":'category' | "SubProduct":'category' | "Narrative":'string' | "Company":'category'
        MASK_Names -- set object containing MASK names applied over Narrative
        
    INPLACE MODIFY :
        <NA>
    
    """
    Data_Narrative = pd.read_csv(Source_file,
        header=0, index_col=False, 
        usecols=["Date received","Product","Sub-product","Consumer complaint narrative","Company"],
        parse_dates=["Date received"], # timestamp is system generated ISO8601, no need to clean it up.
        dtype={"Product":'string',"Sub-product":'string',"Consumer complaint narrative":'string',"Company":'string'},
    ).rename(
        columns={"Date received":"DateReceived", "Sub-product":"SubProduct", "Consumer complaint narrative":"Narrative"}
    ).dropna(
        subset=["Product","Narrative"], ignore_index=True
        # Tag is not manually typed but drop-down options. Blank narratives are already filtered out at the server side. 
        # Very unlikely to have missing values; the step is only for safety measure.
    ) 
    Data_Narrative["DateReceived"] = Data_Narrative["DateReceived"].dt.to_period('D') # keep this column, for trend analysis later
    for col in ["Product","SubProduct","Company"] : # 'category' dtype is much more efficient when n_unique<<n 
        Data_Narrative[col] = Data_Narrative[col].str.lower().str.strip().str.replace(r'[^a-z0-9]+','_',regex=True).astype('category')  
    s = Data_Narrative["Narrative"].str.strip()
    # noticed a small fraction of records storing byte-string b"..." as literal string in the source csv
    # although not a significant chunk, still evaluating those records can give a cleaner text by removing common escape sequence like \n \t \' etc   
    def bstr_to_str(x) :
        try : return ast.literal_eval(x).decode('utf-8')
        except Exception : return x
    bstr_mask = (s.str.startswith('b"',na=False) & s.str.endswith('"',na=False))|(s.str.startswith('b\'',na=False) & s.str.endswith('\'',na=False))    
    s.loc[bstr_mask] = s.loc[bstr_mask].map(bstr_to_str)
    # assign tag to a few common PII_MASK(XX+) patterns observed in the sample narratives, to avoid unpredictable tokenization results. 
    # the tag is not exhaustive, but it should help mitigate the most common cases found at first glance.
    MASK_Collections = {
        r'\bXX/XX/(?:XXXX\b|\s?20\d{2}\b|year[^\s.,]*)':'MASK_DATE',
        r'(?<!\S)\$\s?XXXX\b':'MASK_AMOUNT',
        r'\bX{4,}(?:\s+X{4,})*\b':'MASK_PII' 
        # the ordering of the dict matters 
        # fallback category must be the last item
    } ; MASK_Names = set(MASK_Collections.values()) # for faster lookup in downstream. 
    s = s.str.replace(
        MASK_Collections | {r'\s+':' '}, # tag common masks and normalize whitespaces  
        regex=True
    )
    Data_Narrative["Narrative"] = s

    return Data_Narrative,MASK_Names



if __name__=='__main__' : 
    pass
