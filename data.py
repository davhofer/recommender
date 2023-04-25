import torch
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import numpy as np

def preprocess_transactions(df):
        
    df = df[~df['topic_id'].isna()]
    df['topic_id'] = pd.to_numeric(df['topic_id'])

    # only keep users with at least 10 interactions
    df = df.groupby('user_id').filter(lambda x: len(x) >= 10)

    # all user_ids in the dataset
    user_ids = list(df['user_id'].unique())

    # all topic_ids in the dataset
    topic_ids = list(df['topic_id'].unique())

    # train test split
    df['start_time'] = pd.to_datetime(df['start_time'])

    cut = np.percentile(df['start_time'], 80)

    train_df = df[df['start_time'] < cut]
    test_df = df[df['start_time'] >= cut]

    return train_df, test_df, user_ids, topic_ids




class TransactionsStudentsTopicsDS(Dataset):
    def __init__(self, df, user_ids, topic_ids, negative_frac=1.0):

        self.user_ids = user_ids
        self.topic_ids = topic_ids

     
        interactions = list(df.groupby(['user_id', 'topic_id']).count().index)

        all_pairings = {(user, topic) for user in user_ids for topic in topic_ids}
        positives = set(interactions)
        no_interaction = all_pairings - positives
        negatives = random.sample(list(no_interaction), int(negative_frac*len(positives)))

        self.data = [(x[0], x[1], 1.0) for x in positives] + [(x[0], x[1], 0.0) for x in negatives]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user, topic, y = self.data[index]

        user = self.user_ids.index(user)
        user = torch.tensor(user)

        topic = self.topic_ids.index(topic)
        topic = torch.tensor(topic)

        y = torch.tensor([y])
        return user, topic, y


def get_transactions_dataloader(dataframe, user_ids, topic_ids, batch_size, negative_frac=1.0):
    dataset = TransactionsStudentsTopicsDS(dataframe, user_ids, topic_ids, negative_frac=negative_frac)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)