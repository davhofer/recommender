from collections import defaultdict
import pytorch_lightning as pl
import torch
import math
import numpy as np

# def HitRate(model, eval_one_set, topic_ids, n=10):
#     """
#     @arg:
#     test_data: list of [user, topic] - only positive interaction

#     """
#     hits = 0
#     totals = eval_one_set.shape[0]

#     # Load the model and return probability for each user with all topics
#     # Output self.proba for each user/topic pair
#     for user, topic in eval_one_set:
#         test_set = torch.cat([user] * len(topic_ids), topic_ids)
#         model.test(test_set)
#         proba = model.predict_proba

#         # Get the top N of highest probability and rank them 
#         topN = [x for _, x in sorted(zip(proba, topic_ids), reverse=True)]
#         hit = False
#         for top_topic in topN:
#             if (top_topic == topic):
#                 hit = True
#         if (hit):
#             hits += 1
        
#     print(f"hits/total = {hits}/{totals}")
#     return hits/totals


def HitRate(model, eval_one_set, topic_ids, n=10):
    """
    @args:
    model:
    eval_one_set: list of [user, topic] - only positive interaction
    topic_ids: list of all topics 
    n: keep top n topic recommendation

    @output:
    hit_list: (array) hit[i] == 0/1 indicates hit/miss for test sample i 
    ndcg_list: (array) ndcg[i] score for test sample ith
    """
    hit_list = []
    ndcg_list = []
    
    # Load the model and return probability for each user with all topics
    # Output self.proba for each user/topic pair
    for user, topic in eval_one_set:
        test_set = torch.cat([user] * len(topic_ids), topic_ids)
        model.test(test_set)
        proba = model.predict_proba

        # Get the top N of highest probability and rank them 
        topN = [x for _, x in sorted(zip(proba, topic_ids), reverse=True)][:n]
        
        # Calculate hit rate
        hit_list.append(getHitRatio(topN, topic))
        
        # Calculate NDCG
        ndcg_list.append(getNDCG(topN, topic))
        
    print(f'HR@{n}: {np.array(hit_list).mean()}, NDCG@{n}: {np.array(ndcg_list).mean()}')
    return (hit_list, ndcg_list)


def getHitRatio(ranklist, topic):
    for item in ranklist:
        if item == topic:
            return 1
    return 0

def getNDCG(ranklist, topic):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == topic:
            return math.log(2) / math.log(i+2)
    return 0
