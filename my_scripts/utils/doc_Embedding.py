import numpy as np, pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel



MODEL_NAME = 'BAAI/bge-base-en-v1.5'
DEVICE = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
WINDOW_SIZE = tokenizer.model_max_length - tokenizer.num_special_tokens_to_add(pair=False)
OVERLAP = 32 
STRIDE = WINDOW_SIZE - OVERLAP
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
EMBEDDING_DIM = model.config.hidden_size


@torch.inference_mode()
def Embedding__bge_base(X:pd.Series,batch_size:int=32) -> np.ndarray :
    """
    This wrapper function helps us to keep the heavy sentence embedding layer separate from BERTopic topic modeling,
    and reuse same embedding matrix across multiple models.

    Workflow
    --------
    1. Tokenize all the documents.
    2. Split long documents into overlapping token windows.
    3. Batch chunk inference on the GPU.
    4. Extract the CLS embedding from each chunk.
    5. Weighted-average chunk embeddings back to document embeddings.
    6. L2-normalize the final document embeddings.

    [
        We use `BAAI/bge-base-en-v1.5` which has better MTEB Clustering Score and 2x Embedding Dimensions compared to default `all-MiniLM-L6-v2`,
        while still fit well in consumer GPU.

        We use a sliding-window with overlap to avoid truncation loss. 
    ]

    INPUT : 
        X -- pandas.Series[str]
            Narratives 
        batch_size -- int ; Default 32
            batch_size for GPU forward pass

    RETURN : 
        numpy.ndarray -- shape (n_documents,embedding_dimension)
            
    INPLACE MODIFY :
        <NA>
     
    """
    if X.empty : return np.empty((0,EMBEDDING_DIM),dtype=np.float32)
    n = len(X)

    ## ==== Tokenizing & Chunking =====================
    X_tokenized = tokenizer(
        X.fillna('').tolist(),
        add_special_tokens=False,padding=False,truncation=False
    )["input_ids"] # tokenize all docs at one shot
    chunk_source_id,chunk_tokens,chunk_weights = [],[],[]
    for doc_id,tokenized_text in enumerate(X_tokenized) :
        # each iteration split a document in a list of chunks
        doc_len = len(tokenized_text)
        if doc_len>0 :
            token_idx = 0
            while True :
                chunk = tokenized_text[token_idx : token_idx+WINDOW_SIZE] 
                chunk_source_id.append(doc_id)
                # while chunking a specific document, 
                # store its doc_id for future reference
                chunk_weights.append(len(chunk)) 
                chunk_tokens.append(
                    {
                        "input_ids":tokenizer.build_inputs_with_special_tokens(chunk)
                    } # add [CLS],[SEP] 
                )
                if token_idx+WINDOW_SIZE < doc_len : 
                    token_idx += STRIDE
                else : break
        else : 
            # Handle empty documents by adding an empty chunk to preserve indices
            chunk_source_id.append(doc_id)
            chunk_weights.append(1.0) # avoids 0-division
            chunk_tokens.append({"input_ids":[tokenizer.cls_token_id,tokenizer.sep_token_id]})
    # multiple rows of chunk_tokens may correspond to same doc_id 
    chunk_source_id = torch.tensor(chunk_source_id,dtype=torch.long,device=DEVICE)
    chunk_weights = torch.tensor(chunk_weights,dtype=torch.float32,device=DEVICE).unsqueeze(1)
    ##  ----- ---------------

    ## ==== Batch Processing =====================
    # Instead of using SentenceTransformer over text, we directly process tokens with AutoModel
    # It avoids Tokenize(doc)-->Chunking-->DetokenizeChunks-->Process(Chunk text) redundancy
    # But require us to implement TokenEmbedding--> Pooling--> SentenceEmbedding by our own
    Embedding_Agg = torch.zeros((n,EMBEDDING_DIM),dtype=torch.float32,device=DEVICE) # placeholder

    for batch_idx in range(0,len(chunk_tokens),batch_size) : 
        batch_slice = slice(batch_idx,batch_idx+batch_size)
        batch = tokenizer.pad(
            chunk_tokens[batch_slice], # pass multiple chunks at a time for better GPU utilization
            padding=True, # ensure all items in a batch equal length, via padding
            return_tensors='pt' 
        ).to(DEVICE) 

        with torch.autocast(device_type='cuda',dtype=torch.bfloat16) : 
            # The actual forward pass happen here
            outputs = model(**batch) 
        
        chunk_embeddings = F.normalize(
            outputs.last_hidden_state[:,0].to(torch.float32), # BGE is designed for CLS-pooling
            p=2,dim=1
        ) # BGE is designed for cosine similarity
        Embedding_Agg.index_add_(0,chunk_source_id[batch_slice],chunk_embeddings.mul_(chunk_weights[batch_slice]))
        # each row of Embedding_Agg holds the weighted sum of token embeddings in corresponding doc_id
        # each weights ~1.0, except the last window of a doc which may not span the full WINDOW_SIZE
    ##  ----- ---------------

    document_embeddings = F.normalize(
        Embedding_Agg, 
        # L2 normalization on sum() or mean() gives same results
        p=2,dim=1
    )

    return document_embeddings.cpu().numpy()