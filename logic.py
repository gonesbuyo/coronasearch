import pandas as pd
import numpy as np
import faiss
import pickle

#import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer

"""
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel
import torch

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
"""

def load_data():
    # returns a df with the data columns currently used
    data_path = 'resources/cord19_data'
    return pd.read_pickle(data_path)[['index', 'title', 'cord_uid', 'doi', 'source_x', 
                                       'pmcid', 'pubmed_id', 'license', 'abstract', 
                                       'publish_time', 'authors', 'journal', 'url']]

def load_index():

    """
    with open('coronasearch/resources/index.pkl', 'rb') as index:
        embeddings_matrix = np.array([pickle.load(index) for i in range(n_docs)])
        
    return embeddings_matrix
    """

    # returns the docs index
    return faiss.read_index('resources/faiss_index')

def load_muse():
    model = hub.load('resources/muse/')

    return model

def get_muse_query(input_text, model):
    query = model(input_text).numpy()

    return query

def load_bert():
    # returns DistilmBERT tokenizer and model
    config = DistilBertConfig.from_pretrained('resources/coronabert/', 
                                              output_hidden_states=True)

    tokenizer = DistilBertTokenizer.from_pretrained('resources/coronabert/', 
                                                    config=config)

    model = DistilBertModel.from_pretrained('resources/coronabert/', 
                                            config=config)
    model.eval()

    return tokenizer, model

def get_query(input_text, tokenizer, model):
    query = tokenizer.tokenize(input_text)
    #print(query)
    
    gensim_dict = Dictionary.load_from_text('resources/gensim_dict.txt')
    tfidf_model = TfidfModel.load('resources/tfidf.pkl')
    
    query_bow = gensim_dict.doc2bow(query)
    q_tfidfs = tfidf_model[query_bow]
    #print(q_tfidfs)
    
    # tfidf dict for the query
    query_tfidfs = {e[0]: e[1] for e in q_tfidfs}
    # convert query tokens to gensim indices 
    gensim_query = gensim_dict.doc2idx(query)

    # from indices to tfidf scores
    query_tfidf_list = [query_tfidfs[token] if token in query_tfidfs else 0 for token in gensim_query]
    #print(query_tfidf_list)
    
    # turn list into tensor
    query_tfidf_tensor = torch.tensor(query_tfidf_list).unsqueeze(0)
    
    query_tokens = tokenizer.convert_tokens_to_ids(query)
    
    input_tensor = torch.tensor([query_tokens])
    with torch.no_grad():
        bert_output = model(input_tensor)[1]

    #query_raw_embeddings = torch.mean(torch.stack(bert_output), dim=0)
    query_raw_embeddings = bert_output[5]

    query_tokens_embedding = query_raw_embeddings*query_tfidf_tensor.unsqueeze(2)

    # Get a dict with the sents and positions (tuple array) where each token is located
    token_positions = {}
    n_tokens = 0 # number of tokens in the query
    for n_token, token_id in enumerate(query_tokens):
        if token_id in token_positions:
            token_positions[token_id].append(n_token)
        else:
            token_positions[token_id] = [n_token]
        n_tokens += 1

    # Get a list with one embedding for each token (by averaging the ones with more than one)
    tensors_list = []
    for token_key, positions in token_positions.items():
        token_embedding_list = []
        for pos in positions:
            token_embedding_list.append(query_tokens_embedding[0][pos])
        #token_embedding = torch.mean(torch.stack(token_embedding_list), dim=0)
        token_embedding = torch.sum(torch.stack(token_embedding_list), dim=0)
        tensors_list.append(token_embedding)
        
    # Create a tensor with the resulting token embeddings and apply sigmoid function
    #sigmoid = torch.nn.Sigmoid()
    #embeddings_tensor = sigmoid(torch.stack(tensors_list))
    embeddings_tensor = torch.stack(tensors_list)

    # Apply mean tfidf pooling and softmax to the embeddings tensor
    softmax = torch.nn.Softmax(dim=0)
    query_embedding = softmax((torch.sum(embeddings_tensor, dim=0))/n_tokens).detach().numpy()
    #print(query_embedding)

    return query_embedding

def get_results(query_embedding, index, n_results):

    # returns indices of results
    # as of now the system handles one query each time
    #single_query = np.expand_dims(query_embedding, axis=0)
    single_query = query_embedding
    # 0 is distances and 1 indices, we only want indices
    indices = index.search(single_query, n_results)[1]

    """
    P = index
    Q = query_embedding
    kl_divergences = (P * np.log(P / Q)).sum(axis=1)
    
    return np.argsort(kl_divergences)[:n_results].tolist()
    """

    return indices[0]
