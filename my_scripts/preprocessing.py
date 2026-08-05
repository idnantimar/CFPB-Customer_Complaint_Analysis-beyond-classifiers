import pandas as pd
import ast
import truecase
import re


def clean_text(Source_dir) -> tuple[pd.DataFrame,frozenset[str]] :
    """ 
    Read the raw CFPB data from a given path and return it with cleaned and trimmed data.

    INPUT : 
        Source_dir -- glob.glob("*.csv") for pandas.read_csv(...)
        
    RETURN : 
        Data_Narrative -- pandas.DataFrame 
            | "Complaint ID":Index || "DateReceived":'period[D]' | "Product":'category' | "SubProduct":'category' | "Issue":'category' | "SubIssue":'category' | "Narrative":'string' | "Company":'category' | "Response":'category' |
            Sorted in ascending order of "DateReceived"
        MASK_Names -- frozenset containing the canonical MASK tokens applied to the narratives
           
    """
    Data_Narrative = pd.concat(
        [
            pd.read_csv(file,
                header=0, index_col=False, dtype='string[pyarrow]',
                usecols=[
                    "Date received","Complaint ID",
                    "Product","Sub-product","Issue","Sub-issue",
                    "Consumer complaint narrative",
                    "Company","Company response to consumer"
                ],
                parse_dates=["Date received"], # timestamp is system generated ISO8601
            ) for file in Source_dir 
        ], axis=0, ignore_index=True
    ).rename(
        columns={
            "Date received":"DateReceived", 
            "Complaint ID":"ComplaintID",
            "Sub-product":"SubProduct", "Sub-issue":"SubIssue",
            "Consumer complaint narrative":"Narrative",
            "Company response to consumer":"Response"
        }
    )
    ## === REMOVE NULL ====
    NA_mask = Data_Narrative[["Product","Narrative"]].isna().to_numpy()
    print(f'''HasNA('Product','Narrative'):{NA_mask.sum(dtype=int)}''')
    if NA_mask.any() : 
        Data_Narrative = Data_Narrative.dropna(
            axis=0, subset=["Product","Narrative"], ignore_index=True
            # Tag is not manually typed but drop-down options. Blank narratives are already filtered out at the server side.
            # Very unlikely to have missing values; the step is only for safety measure.
        )
    ## --- ---
    ## === SORT BY DATE ====
    # stable sort to ensure row order across different runs of the script, for multiple complaints within same date 
    Data_Narrative["DateReceived"] = Data_Narrative["DateReceived"].dt.to_period('D') 
    Data_Narrative = Data_Narrative.sort_values(by="DateReceived",ignore_index=True,kind='stable') 
    ## --- ---
    ## === CATEGORICAL COLUMNS ====
    # 'category' dtype is much more efficient when n_unique<<n 
    for col in ["Product","SubProduct","Issue","SubIssue","Company","Response"] :
        Data_Narrative[col] = Data_Narrative[col].str.lower().str.strip().str.replace(r'[^a-z0-9]+','_',regex=True).astype('category')  
    ## --- ---
    ## === CLEAN TEXT ====
    s = Data_Narrative["Narrative"].str.strip()
    # noticed a small fraction of records storing byte-string b"..." as literal string in the source csv
    # although not a significant chunk, still evaluating those records can give a cleaner text by removing common escape sequence like \n \t \' etc   
    def bstr_to_str(x) :
        try : return ast.literal_eval(x).decode('utf-8')
        except Exception : return x
    bstr_mask = (s.str.startswith('b"',na=False) & s.str.endswith('"',na=False))|(s.str.startswith('b\'',na=False) & s.str.endswith('\'',na=False))    
    s.loc[bstr_mask] = s.loc[bstr_mask].map(bstr_to_str)
    # a small fraction of narratives are written almost entirely in uppercase/lowercase. 
    # Apply statistical truecasing only to these narratives, leaving naturally cased text unchanged. 
    # This can improve downstream performance of case-sensitive embedding models (e.g., DeBERTa, BGE). 
    n_letters = s.str.count(r'[A-Za-z]').fillna(0).clip(1,None)
    upper_rate = s.str.count(r'[A-Z]').fillna(0)/n_letters
    review_casing = ((upper_rate>0.8) | (upper_rate<0.001)) & (n_letters>10)
    s.loc[review_casing] = [truecase.get_true_case(x) for x in s.loc[review_casing].str.lower().to_list()]
    # assign tag to a few common PII_MASK(XX+) patterns observed in the sample narratives, to avoid unpredictable tokenization results 
    # the tag is not exhaustive, but it should help mitigate the most common cases found at first glance
    for rules in MASK_COLLECTIONS.values():
        for pat,repl in rules:
            while True:
                new = (
                    s.str.replace(
                        pat=pat,repl=repl,
                        regex=True
                    )
                )
                not_changed = new.equals(s)
                s = new
                if not_changed : break
    Data_Narrative["Narrative"] = (
        # whitespace normalized
        s
        .str.replace(r'\s+',' ',regex=True)
        .str.replace(r'\s+([.,:;!?])',r'\1',regex=True)
        .str.strip()
    )
    ## --- ---
    ## === UNIQUE ID ====
    # Every complaint record has a unique ID; this step is only for safety measure.
    Data_Narrative["ComplaintID"] = Data_Narrative["ComplaintID"].str.strip()
    Data_Narrative = Data_Narrative.drop_duplicates(subset=["ComplaintID"],keep='last',ignore_index=True).set_index("ComplaintID")
    # We should not remove duplicates by narrative, even if two complaint narratives are identical.
    # Repeated complaints indicate that the corresponding complaint theme is common and should be reflected in the analysis.
    ## --- ---
    return Data_Narrative,frozenset(MASK_COLLECTIONS)



MASK_COLLECTIONS = {
    'MASK_DATE': [
        (
                re.compile(r'([a-z0-9]*)(XX\s*/\s*XX\s*/\s*(?:XXXX|\d{1,4}|(?:year|scrub)[^a-z0-9\s.,:;]*))([a-z0-9]*)',flags=re.IGNORECASE),
                r'\1 MASK_DATE \3'
        ),
        (
                re.compile(r'([a-z0-9]*)(XX\s*/\s*XX\s*/)([a-z0-9]*)',flags=re.IGNORECASE),
                r'\1 MASK_DATE \3'
        )
    ],
    'MASK_AMOUNT': [
        (
            re.compile(r'([a-z0-9]*)(\$\s?X{3,})([a-wyz0-9]*)',flags=re.IGNORECASE),
            r'\1 MASK_AMOUNT \3'
        )
    ],
    'MASK_PII': [
        (
            re.compile(r'([a-wyz0-9]*)(X{3,}(?:\s+X{3,})*)([a-wyz0-9]*)',flags=re.IGNORECASE),
            # if the mask blends with neighbor word starting or ending with x (e.g. AmexXXXX) it will return Ame MASK_PII
            # this is a relatively rare scenario and we accept it than accepting high miss-out in using strict word-boundary based replacement 
            r'\1 MASK_PII \3'
        )
    ],
    # tag common masks 
    # case-insensitive matching also handles the occasional lowercase mask
    # the extra space prevents masks from blending with immediate neighbors
    # space can later be normalized easily
}