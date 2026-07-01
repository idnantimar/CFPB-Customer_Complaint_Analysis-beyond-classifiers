import numpy as np, pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel
from tqdm.auto import tqdm



MODEL_NAME = 'BAAI/bge-base-en-v1.5'
DEVICE = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
WINDOW_SIZE = tokenizer.model_max_length - 2
CLS_ID,SEP_ID = tokenizer.cls_token_id,tokenizer.sep_token_id
OVERLAP = 32 
STRIDE = WINDOW_SIZE - OVERLAP
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
EMBEDDING_DIM = model.config.hidden_size


@torch.inference_mode()
def Embedding__bge_base(X:pd.Series,*,batch_size:int=32) -> np.ndarray :
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
    print("Tokenization : SUCCESS")
    chunk_source_id,chunk_start_idx,chunk_end_idx = [],[],[]
    for doc_id,doc_len in enumerate(map(len,X_tokenized)) :
        # each iteration split a document in a list of chunks
        if 0 < doc_len <= WINDOW_SIZE :
            # most of the documents fall in this category where no calculation required
            # we avoid the extra overhead of array.extend() compared to array.append()
            chunk_source_id.append(doc_id)
            chunk_start_idx.append(0)
            chunk_end_idx.append(doc_len)            
        elif doc_len > WINDOW_SIZE :
            # Heuristic : if Last_Position_Current_Window >= Last_Position_Document, we have covered the whole document
            # i.e. (start_position+WINDOW_SIZE-1) >= (doc_len-1) ==> start_position >= (doc_len-WINDOW_SIZE) ==> it is the last window  
            # Example : array = [0,1,2,3,...,94,95] ; WINDOW_SIZE=40, STRIDE=30, OVERLAP=10
            #   1st Chunk [0,1,2,...,39]
            #   2nd Chunk [30,31,32,...,69]
            #   3rd Chunk [60,61,62,...,94,95] ---> this is last ;  we don't require another chunk of [90,91,...,94,95]
            start_position = np.arange(0,doc_len-WINDOW_SIZE,STRIDE) ; start_position = np.concatenate((start_position,[start_position[-1]+STRIDE]))
            end_position = start_position+WINDOW_SIZE ; end_position[-1] = doc_len
            chunk_start_idx.extend(start_position)
            chunk_end_idx.extend(end_position)
            chunk_source_id.extend([doc_id]*len(start_position))
        else : 
            # Handle empty documents by adding an empty chunk to preserve indices
            chunk_source_id.append(doc_id)
            chunk_start_idx.append(0)
            chunk_end_idx.append(0)
    print("Chunking : SUCCESS")
    # multiple chunks correspond to same doc_id, chunk_source_id keep track of the doc_id per row for future referance
    # [chunk_start_idx,chunk_end_idx) slice a chunk out the entire text of X_tokenized.iloc[chunk_source_id]
    chunk_source_id = np.array(chunk_source_id,dtype=np.int32)
    chunk_start_idx,chunk_end_idx = np.array(chunk_start_idx,dtype=np.int32),np.array(chunk_end_idx,dtype=np.int32)
    ##  ----- ---------------

    ## ==== Batch Processing =====================
    # Instead of using SentenceTransformer over text, we directly process tokens with AutoModel
    # It avoids Tokenize(doc)-->Chunking-->DetokenizeChunks-->Process(Chunk text) redundancy
    # But require us to implement TokenEmbedding--> Pooling--> SentenceEmbedding by our own
    Embedding_Agg = torch.zeros((n,EMBEDDING_DIM),dtype=torch.float32,device=DEVICE) # placeholder
    chunk_weights = torch.as_tensor(chunk_end_idx-chunk_start_idx,dtype=torch.float32,device=DEVICE).clip(min=1.0).unsqueeze(-1) 
    # len of every chunk represents its relative importance when aggregating multiple chunk embedding of same document 
    # Heuristic : BGE models are designed for CLS-pooling
    #   Here CLS vector represents its entire context-window, capturing interactions across all tokens.
    #   In sliding window setup, every chunk contributes proportional to the amount of context it holds.
    chunk_source_id_gpu = torch.as_tensor(chunk_source_id,dtype=torch.long,device=DEVICE) # same data ; avoid repeated CPU-->GPU conversion in a loop
    n_chunks = len(chunk_source_id)
    print("Batch Processing : IN PROGRESS")
    n_batches = (n_chunks+batch_size-1)//batch_size
    for batch_idx in tqdm(range(0,n_chunks,batch_size),total=n_batches,desc="Embedding documents",unit="batch") : 
        batch_slice = slice(batch_idx,min(batch_idx+batch_size,n_chunks))
        batch_tokens = [
            [
                CLS_ID,
                *((X_tokenized[chunk_source_id[r]])[chunk_start_idx[r]:chunk_end_idx[r]]),
                SEP_ID
            ] for r in range(batch_slice.start,batch_slice.stop)
        ]
        batch = tokenizer.pad(
            {"input_ids":batch_tokens}, # pass multiple chunks at a time for better GPU utilization
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
        # dimension(chunk_embeddings) = (batch_size,EMBEDDING_DIM)
        Embedding_Agg.index_add_(0,chunk_source_id_gpu[batch_slice],chunk_embeddings.mul_(chunk_weights[batch_slice]))
        # each row of Embedding_Agg holds the weighted sum of chunk embeddings in corresponding doc_id
    print("Batch Processing : SUCCESS")
    ##  ----- ---------------
    print(f"Returning Final numpy.ndarray shape=({(n,EMBEDDING_DIM)}) ...")
    document_embeddings = F.normalize(
        Embedding_Agg, 
        # L2 normalization on sum() or mean() gives same results
        p=2,dim=1
    )

    return document_embeddings.cpu().numpy()