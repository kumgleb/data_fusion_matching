import pickle


def load_code_to_idx(mcc_code_to_idx_path, cat_code_to_idx_path):
    with open(mcc_code_to_idx_path, "rb") as f:
        mcc_code_to_idx = pickle.load(f)

    with open(cat_code_to_idx_path, "rb") as f:
        cat_code_to_idx = pickle.load(f)

    return mcc_code_to_idx, cat_code_to_idx