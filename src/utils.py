import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from typing import Dict


#  calc_distances, get_words_and_embs, get_top_k_words, filter_mcc_descriptipn

def get_corpus(clickstream_categories, stopwords):
    corpus = set()
    for idx in clickstream_categories.index:
        descr = clickstream_categories.loc[idx, "Description"]
        for w in descr.split(" "):
            if w not in stopwords:
                corpus.add(w)
    return list(corpus)


def get_embs(descr, wv_embeddings, stopwords):
    embs = []
    words = []
    for w in descr.split(" "):
        if w not in stopwords and w not in words:
            try:
                emb = wv_embeddings[w]
                embs.append(emb)
                words.append(w)
            except KeyError:
                continue
    return words, np.array(embs)


def get_top_k_words(mcc_descriptipn, clck_corpus_embs, wv_embeddings, stopwords, K):

    words, embs = get_embs(mcc_descriptipn, wv_embeddings, stopwords)
    distances = embs[np.newaxis, :, :] - clck_corpus_embs[:, np.newaxis, :]
    distances = np.linalg.norm(distances, axis=-1)
    min_distances = np.min(distances, axis=0)

    words_dists = [(w, dist) for w, dist in zip(words, min_distances)]
    words_dists.sort(key=lambda x: x[1], reverse=False)

    top_k = [w[0] for w in words_dists[:K]]

    return top_k


def create_text_embed(discr, wv_embeddings):
    pass


def filter_mcc_descriptipn(mcc_codes, mcc_descriptipn, clck_corpus_embs, wv_embeddings, stopwords, k=1):

    for idx in tqdm(mcc_codes.index):
        mcc_descriptipn = mcc_codes.loc[idx, "Description"]
        top_k = get_top_k_words(mcc_descriptipn, clck_corpus_embs, wv_embeddings, stopwords, k)
        mcc_codes.loc[idx, "Description"] = (" ").join(top_k)

    return mcc_codes



def create_clck_embeddings_dict(
    clickstream_cat: pd.DataFrame, wv_embeddings: KeyedVectors, mode="WE", model=None
) -> Dict:

    embs = {}

    for idx in tqdm(clickstream_cat.index):
        clck_code = clickstream_cat.loc[idx, "cat_id"]
        discr = clickstream_cat.loc[idx, "Description"]

    
        if mode == "WE":
            embs[clck_code] = create_text_embed(discr, wv_embeddings)
        elif mode == "ST":
            embs[clck_code] = model.encode(discr)


    with open("./embeddings/new_clck_cat_emb_en_filtered.pickle", "wb") as f:
        pickle.dump(embs, f, protocol=pickle.DEFAULT_PROTOCOL)

    return embs

