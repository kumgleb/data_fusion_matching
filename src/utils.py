

# get_corpus, calc_distances, get_words_and_embs, get_top_k_words, filter_mcc_descriptipn



def filter_mcc_descriptipn(mcc_codes, mcc_descriptipn, clck_corpus_embs, wv_embeddings, stopwords, k=1):

    for idx in tqdm(mcc_codes.index):
        mcc_descriptipn = mcc_codes.loc[idx, "Description"]
        top_k = get_top_k_words(mcc_descriptipn, clck_corpus_embs, wv_embeddings, stopwords, k)
        mcc_codes.loc[idx, "Description"] = (" ").join(top_k)

    return mcc_codes



def create_clck_embeddings_dict(
    clickstream_cat: pd.DataFrame, wv_embeddings: KeyedVectors
) -> Dict:

    embs = {}

    for idx in tqdm(clickstream_cat.index):
        clck_code = clickstream_cat.loc[idx, "cat_id"]
        discr = clickstream_cat.loc[idx, "Description"]

    
        if MODE == "WE":
            embs[clck_code] = create_text_embed(discr, wv_embeddings)
        elif MODE == "ST":
            embs[clck_code] = model.encode(discr)


    with open("./embeddings/new_clck_cat_emb_en_filtered.pickle", "wb") as f:
        pickle.dump(embs, f, protocol=pickle.DEFAULT_PROTOCOL)

    return embs

