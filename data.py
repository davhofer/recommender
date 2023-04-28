import torch
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import numpy as np

def preprocess_events(df):
    df = df[~df['topic_id'].isna()]
    df = df[~df['session_id'].isna()]

    df['topic_id'] = df['topic_id'].astype(int)
    df['session_id'] = df['session_id'].astype(int)
    df['event_date'] = pd.to_datetime(df['event_date'])

    df['action'] = df['action'].replace('VIEW_QUESTION', 'tmp')
    df['action'] = df['action'].replace('REVIEW_TASK', 'VIEW_QUESTION')
    df['action'] = df['action'].replace('tmp', 'REVIEW_TASK')


    def first(x):
        return list(x)[0]

    aggregators = {
        'event_date': 'max',
        'action': set,
        'topic_id': first,
        'category': set,
        'event_type': set,
        'session_type': first,
        'session_closed': first,
        'session_accepted': first,
        'tracking_data': list, #lambda x: list(filter(lambda y: isinstance(y, dict), x)),
    }

    interesting_cols = ['user_id', 'session_id'] + list(aggregators.keys())

    other_aggs = {}
    for c in df.columns:
        if c not in ['user_id', 'session_id'] and c not in aggregators.keys():
            other_aggs[c] = lambda x: 0



    aggregators.update(other_aggs)

    df = df.groupby(['user_id', 'session_id']).agg(aggregators).reset_index()


    filter_users = df[['user_id', 'session_id']].groupby('user_id').nunique()
    filter_users = filter_users[filter_users['session_id'] >= 5]
    filter_users = filter_users.index


    df = df[df['user_id'].isin(filter_users)]

    df = df[interesting_cols]

    return df

class LeaveOneOutDS(Dataset):
    def __init__(self, data, user_ids, topic_ids):
        self.data = data
        self.user_ids = user_ids
        self.topic_ids = topic_ids

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


"""
Testing Procedure:
test set consists of a single interaction for each user (or a subset of users)
that was removed from the training set.
"""

class LeaveOneOutSplitter:
    def __init__(self,
                 df,
                 test_user_frac=0.5,
                 train_negative_frac=1.0,
                 test_sample_strat="newest",
                 ):
        if test_sample_strat not in ['newest', 'random']:
            print("'test_sample_strat' should either be 'newest' or 'random'!")
            return
        
        self.df = df 
        self.train_negative_frac = train_negative_frac
        self.test_sample_strat = test_sample_strat

        self.user_ids = list(df['user_id'].unique())
        self.topic_ids = list(df['topic_id'].unique())

        self.num_students = len(self.user_ids)
        self.num_topics = len(self.topic_ids)

        interactions = list(df.groupby(['user_id', 'topic_id']).count().index)

        all_pairings = {(user, topic) for user in self.user_ids for topic in self.topic_ids}
        positives = set(interactions)
        no_interaction = all_pairings - positives

        test_size = int(test_user_frac * len(self.user_ids))

        test_samples = []
        if self.test_sample_strat == 'random':
            for id in self.user_ids:
                user_interactions = list(filter(lambda x: x[0] == id, interactions))
                s = random.choice(user_interactions)
                test_samples.append(s)
        else:
            user_last_event = df[['user_id', 'event_date']].groupby('user_id').max()
            df['test_set'] = df.apply(lambda row: row['event_date'] == user_last_event['event_date'][row['user_id']], axis=1)
            test_samples = list(df[df['test_set']].groupby(['user_id', 'topic_id']).count().index)
        
        test_samples = random.sample(test_samples, test_size)

        self.test_data = []
        for user_id, topic_id in test_samples:
            for t in self.topic_ids:
                label = 1.0 if topic_id == t else 0.0
                self.test_data.append((user_id, t, label))


        for s in test_samples:
            interactions.remove(s)


        positives = set(interactions)
        negatives = random.sample(list(no_interaction), int(train_negative_frac*len(positives)))

        self.data = [(x[0], x[1], 1.0) for x in positives] + [(x[0], x[1], 0.0) for x in negatives]

    def get_data(self):
        return self.data

    def get_test_data(self):
        return self.test_data

    def get_num_students(self):
        return self.num_students

    def get_num_topics(self):
        return self.num_topics

    def get_user_ids(self):
        return self.user_ids

    def get_topic_ids(self):
        return self.topic_ids

    def get_train_dataset(self):
        return LeaveOneOutDS(self.get_data(), self.get_user_ids(), self.get_topic_ids())

    def get_test_dataset(self):
        return LeaveOneOutDS(self.get_test_data(), self.get_user_ids(), self.get_topic_ids())
