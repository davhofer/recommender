import math
import numpy as np
import pandas as pd


def HitRate_NDCG_MRR(df, n):
    hit_list = []
    ndcg_list = []
    mrr_list = []

    user_predict = df.groupby(['user_id'])

    for user, topic in user_predict:
        # Get the top N of highest probability and rank them 
        if (topic['was_interaction']==1).any() == False:
            continue
        topN = [x for _, x in sorted(zip(topic['predict_proba'], topic['topic_id']), reverse=True)][:n]
        
        positive_topic = int(topic[topic['was_interaction']==1]['topic_id'])
        # Calculate hit rate
        hit_list.append(getHitRatio(topN, positive_topic))
        
        # Calculate NDCG
        ndcg_list.append(getNDCG(topN, positive_topic))

        # Calculate MRR
        mrr_list.append(getMRR(topN, positive_topic))
    
    return {f'HitRate@{n}': np.array(hit_list).mean(), f'NDCG@{n}': np.array(ndcg_list).mean(), f'MRR@{n}': np.array(mrr_list).mean()}



def metrics_per_topic(df, n, math_ids=[], german_ids=[]):
    

    metrics = dict()
    math_df = df[df['topic_id'].isin(math_ids)].copy()


    if len(math_df) > 0:
        metrics['math'] = HitRate_NDCG_MRR(math_df, n)
        
    
    german_df = df[df['topic_id'].isin(german_ids)].copy()
    

    if len(german_df) > 0:
        
        metrics['german'] = HitRate_NDCG_MRR(german_df, n)
    
    return metrics

def HitRate_NDCG_MRR_from_CSV(proba_all_topic_csv, n=10, math=True, german=True, math_ids=[], german_ids=[]):
    """
    @args:
    proba_all_topic_csv: {user_id, topic_id, was_interaction, predict_proba} csv file return recommendation probability for every topic 
    n: keep top n topic recommendation

    @output:
    hit_list: (array) hit[i] == 0/1 indicates hit/miss for test sample i 
    ndcg_list: (array) ndcg[i] score for test sample ith
    """

    proba_all_topic_df = pd.read_csv(proba_all_topic_csv)
    metrics = metrics_per_topic(proba_all_topic_df, n, math_ids=math_ids, german_ids=german_ids)
    return metrics

def getMRR(ranklist, topic):
    if topic not in ranklist:
        return 0
    return 1.0/(ranklist.index(topic) + 1)

def getHitRatio(ranklist, topic):
    return int(topic in ranklist)

def getNDCG(ranklist, topic):
    if topic not in ranklist:
        return 0
    return math.log(2) / math.log(ranklist.index(topic)+2)


if __name__ == '__main__':
    HitRate_NDCG_MRR_from_CSV('ncf_64_predictive_factors_first_try_outputs.csv', 5)
    