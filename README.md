# Data Fusion Contest 2022. Matching
 
Solution of a  Data Fusion 2022 user matching challenge. <br>
 
## Approach:
1. Create embeddings for category descriptions.
2. Create `user_id` embeddings as weighted category embeddings.
3. Train siamese neural network with triplet loss based on matched and unmatched pairs.
4. Infer `user_id` embeddings from siamese neural network.
5. Rank matches by distance between final embeddings.
 
### Categories embeddings:
1. Translate categories description RU to EN.
2. Normalize categories description.
3. Create word embeddings.
4. For `mcc_codes.csv` description select only top-k worlds closest to `click_categories.csv` corpus.
5. Create final category embeddings as averaged embeddings of description of k-closest words.
### User embeddings:
1. Calculate sum of active categories.
2. Calculate category weights with softmax from non zero categories.
