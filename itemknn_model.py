import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np

# class ItemKNN(pl.LightningModule):

#     def __init__(self, k):
#         super().__init__()

#         self.k = k
         
#         self.predict_proba = torch.Tensor()
        

#     def train(self, matrix):
#         self.matrix = matrix

#         self.topic_similarity = self.matrix.fillna(0).T.corr()
 
#     def forward(self, users, topics):
#         scores = torch.Tensor()
#         for i in len(users):
#             score = self._predict_topic_score(users[i], topics[i], self.k, self.matrix, self.topic_similarity)
#             scores.append(score.item())
#         return scores

#     def _predict_topic_score(self, user_id, topic_id, k, matrix, topic_similarity):
#         selected_user_interactions = matrix[user_id].to_frame('score').dropna().reset_index()
#         selected_topic_similarity = topic_similarity[[topic_id]][topic_id].to_frame('similarity').reset_index()
#         weighted_similarity = pd.merge(left=selected_user_interactions, right=selected_topic_similarity, on='topic_id', how='inner').sort_values('similarity', ascending=False)[:k]
#         predicted_score = round(np.average(weighted_similarity['score'], weights=weighted_similarity['similarity']), 5)
#         return predicted_score
    
#     def test_step(self, batch, batch_idx):
#         users, topics, y = batch 
#         y_proba = self.forward(users, topics)
#         self.predict_proba = torch.cat((self.predict_proba, y_proba.detach().cpu()))
    

class ItemKNN:

    def __init__(self, k):
        super().__init__()

        self.k = k
         
        self.predict_proba = torch.tensor([])
        

    def train(self, matrix):
        self.matrix = matrix

        self.topic_similarity = self.matrix.fillna(0).T.corr()
 
    def forward(self, user, topic):
        score = self._predict_topic_score(user, topic)
        return torch.tensor(score)

    def _predict_topic_score(self, user_id, topic_id):
        selected_user_interactions = self.matrix[user_id].to_frame('score').dropna().reset_index()
        selected_topic_similarity = self.topic_similarity[[topic_id]][topic_id].to_frame('similarity').reset_index()
        weighted_similarity = pd.merge(left=selected_user_interactions, right=selected_topic_similarity, on='topic_id', how='inner').sort_values('similarity', ascending=False)[:self.k]
        if len(weighted_similarity) == 0:
            return 0.0
        predicted_score = round(np.average(weighted_similarity['score'], weights=weighted_similarity['similarity']), 5)
        return predicted_score
    
    def test_step(self, sample):
        users, topics, y = sample 
        y_proba = self.forward(users, topics)
        self.predict_proba = torch.cat((self.predict_proba,torch.tensor([y_proba.detach().cpu()])))
    
