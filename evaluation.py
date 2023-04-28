import math
import numpy as np
import pandas as pd

def HitRate_NDCG(proba_all_topic_csv, n=10):
    """
    @args:
    proba_all_topic_csv: {user_id, topic_id, was_interaction, predict_proba} csv file return recommendation probability for every topic 
    n: keep top n topic recommendation

    @output:
    hit_list: (array) hit[i] == 0/1 indicates hit/miss for test sample i 
    ndcg_list: (array) ndcg[i] score for test sample ith
    """
    hit_list = []
    ndcg_list = []
    proba_all_topic_df = pd.read_csv(proba_all_topic_csv)
    user_predict = proba_all_topic_df.groupby(['user_id'])

    for user, topic in user_predict:
        #print(topic)
        # Get the top N of highest probability and rank them 
        topN = [x for _, x in sorted(zip(topic['predict_proba'], topic['topic_id']), reverse=True)][:n]
        positive_topic = int(topic[topic['was_interaction']==1]['topic_id'])
        # Calculate hit rate
        hit_list.append(getHitRatio(topN, positive_topic))
        
        # Calculate NDCG
        ndcg_list.append(getNDCG(topN, positive_topic))
        
    return np.array(hit_list).mean(), np.array(ndcg_list).mean()


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
if __name__ == '__main__':
    HitRate('ncf_64_predictive_factors_first_try_outputs.csv', 5)
    