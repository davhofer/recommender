import torch
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import numpy as np


def preprocess_events(df, topics):
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

    topics = topics[['id', 'math']].rename(columns={'id': 'topic_id', 'math': 'is_math'})
    df = pd.merge(left=df, right=topics, on='topic_id')

    return df


def df_add_data(df, col_name, data, fill_na_value=0, key='user_id'):
    dtype = type(data.keys()[0])
    def get_val(row):
        if dtype(row[key]) in data.keys():
            if np.isnan(data[dtype(row[key])]):
                return fill_na_value
            else:
                return data[dtype(row[key])]
        else:
            return fill_na_value
    df[col_name] = df.apply(get_val, axis=1)
    return df

def create_user_features(users, transactions):

    user_features = users[['user_id', 'gender', 'canton', 'class_level']]

    transactions = transactions[~transactions['topic_id'].isna()]
    transactions = transactions[~transactions['user_id'].isna()]

    transactions['transaction_id'] = transactions['transaction_id'].astype(int)
    transactions['user_id'] = transactions['user_id'].astype(int)

    def compute_percentage_correct(transactions_df):

        usrs = pd.DataFrame({'user_id': transactions_df['user_id'].unique(), 'dummy': transactions['user_id'].unique()})
        usrs = usrs.set_index('user_id', drop=True)

        partial_per_user = transactions_df[transactions_df['evaluation'] == 'PARTIAL'].groupby('user_id').count()['transaction_id']
        correct_per_user = transactions_df[transactions_df['evaluation'] == 'CORRECT'].groupby('user_id').count()['transaction_id']
        wrong_per_user = transactions_df[transactions_df['evaluation'] == 'WRONG'].groupby('user_id').count()['transaction_id']

        ppu_keys = partial_per_user.keys()
        cpu_keys = correct_per_user.keys()
        wpu_keys = wrong_per_user.keys()

        def correctness_score(row):
            uid = row.name
            
            n_wrong = 0 if uid not in wpu_keys else wrong_per_user[uid]
            n_partial = 0 if uid not in ppu_keys else partial_per_user[uid]
            n_correct = 0 if uid not in cpu_keys else correct_per_user[uid]

            total = n_wrong + n_correct + n_partial

            score = 100 * (n_correct + 0.5 * n_partial)


            if total == 0:
                return 0
            
            score /= total
            
            return score

        return usrs.apply(correctness_score, axis=1)

    percentage_correct = compute_percentage_correct(transactions)
    transactions = df_add_data(transactions, 'percentage_correct', percentage_correct)


    # features from transactions

    num_topics = transactions[['user_id', 'topic_id']].groupby('user_id').nunique()['topic_id']

    user_features = df_add_data(user_features, 'num_topics', num_topics)

    transactions_per_user = transactions[['transaction_id', 'user_id']]
    transactions_per_user = transactions_per_user.groupby(['user_id']).size()

    user_features = df_add_data(user_features, 'num_transactions', transactions_per_user)

    transactions_per_user_topic = transactions[['transaction_id', 'user_id', 'topic_id']]
    transactions_per_user_topic = transactions_per_user_topic.groupby(['user_id', 'topic_id'])
    per_topic_count = transactions_per_user_topic.count().reset_index()

    per_topic_count = per_topic_count[~per_topic_count['topic_id'].isna()]

    per_topic_mean = per_topic_count.groupby('user_id').mean()['transaction_id']
    per_topic_std = per_topic_count.groupby('user_id').std()['transaction_id']

    user_features = df_add_data(user_features, 'per_topic_mean', per_topic_mean)
    user_features = df_add_data(user_features, 'per_topic_std', per_topic_std)

    avg_performance = transactions[['user_id', 'percentage_correct']].groupby('user_id').mean()['percentage_correct']
    user_features = df_add_data(user_features, 'avg_performance', avg_performance)

    # transform values:
    user_features['gender'] = user_features.apply(lambda x: 1 if x['gender'] == 'MALE' else (2 if x['gender'] == 'FEMALE' else 0), axis=1)
    user_features['canton'].replace(list(user_features['canton'].unique()),
                            list(range(len(user_features['canton'].unique()))), inplace=True)

    user_features['class_level'].replace(list(user_features['class_level'].unique()),
                            list(range(len(user_features['class_level'].unique()))), inplace=True)

    # TODO: replace class level with better encoding, i.e. one-hot/embedding

    #user_features['class_level'].replace(np.nan, 'andere', inplace=True)

    # pd.get_dummies(user_features['class_level'])

    user_features = user_features.set_index('user_id')

    return user_features


def create_topic_features(topics, documents, events):

    events = events[~events['event_id'].isna()]
    events = events[~events['topic_id'].isna()]

    topic_features = topics[['id']].rename(columns={'id': 'topic_id'})

    num_documents = documents[['topic_id', 'document_id']].groupby('topic_id').count()['document_id']
    topic_features = df_add_data(topic_features, 'num_documents', num_documents, key='topic_id')
    num_events = events[['topic_id', 'event_id']].groupby('topic_id').count()['event_id']
    topic_features = df_add_data(topic_features, 'num_events', num_events, key='topic_id')

    topic_features = topic_features.set_index('topic_id')
    return topic_features



class LeaveOneOutDS(Dataset):
    def __init__(self, data, user_ids, topic_ids):
        self.data = data
        self.user_ids = user_ids
        self.topic_ids = topic_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user, topic, features, y = self.data[index]

        user = self.user_ids.index(user)
        user = torch.tensor(user)

        topic = self.topic_ids.index(topic)
        topic = torch.tensor(topic)

        y = torch.tensor([y])
        return user, topic, features, y


"""
Testing Procedure:
test set consists of a single interaction for each user (or a subset of users)
that was removed from the training set.
"""

class LeaveOneOutSplitter:
    def __init__(self,
                 df,
                 use_features=False,
                 user_features=None,
                 topic_features=None,
                 test_user_frac=0.5,
                 val_user_frac=0.5,
                 train_negative_frac=1.0,
                 test_sample_strat="newest",
                 ):
        if test_sample_strat not in ['newest', 'random']:
            print("'test_sample_strat' should either be 'newest' or 'random'!")
            return
        
        self.df = df 
        self.train_negative_frac = train_negative_frac
        self.test_sample_strat = test_sample_strat

        self.use_features = use_features
        self.user_features = user_features
        self.topic_features = topic_features

        if self.use_features:
            if self.user_features is None or self.topic_features is None:
                print("No features have been passed but use_features is True. Init failed!")
                return 
            
            self.num_user_features = self.user_features.shape[1]

            self.num_topic_features = self.topic_features.shape[1]
        
        else:
            self.num_user_features = 0
            self.num_topic_features = 0

        self.user_ids = list(df['user_id'].unique())
        self.topic_ids = list(df['topic_id'].unique())

        self.num_students = len(self.user_ids)
        self.num_topics = len(self.topic_ids)

        interactions = list(df.groupby(['user_id', 'topic_id']).count().index)

        all_pairings = {(user, topic) for user in self.user_ids for topic in self.topic_ids}
        positives = set(interactions)
        no_interaction = all_pairings - positives

        test_size = int(test_user_frac * len(self.user_ids))
        val_size = int(val_user_frac * len(self.user_ids))

        val_test_samples = []
        if self.test_sample_strat == 'random':
            for id in self.user_ids:
                user_interactions = list(filter(lambda x: x[0] == id, interactions))
                s = random.choice(user_interactions)
                val_test_samples.append(s)
        else:
            user_last_event = df[['user_id', 'event_date']].groupby('user_id').max()
            df['test_set'] = df.apply(lambda row: row['event_date'] == user_last_event['event_date'][row['user_id']], axis=1)
            val_test_samples = list(df[df['test_set']].groupby(['user_id', 'topic_id']).count().index)

        test_samples = random.sample(val_test_samples, test_size)
        self.test_data = []
        for user_id, topic_id in test_samples:
            for t in self.topic_ids:
                features = []
                if self.use_features:
                    features.append(self._get_user_feature(user_id))
                    features.append(self._get_topic_feature(t))
                label = 1.0 if topic_id == t else 0.0
                self.test_data.append((user_id, t, features, label))

        for s in self.test_data:
            t = (s[0], s[1])
            if t in val_test_samples:
                val_test_samples.remove((s[0], s[1]))

        val_samples = random.sample(val_test_samples, val_size)
        self.val_data = []
        for user_id, topic_id in val_samples:
            features = []
            if self.use_features:
                features.append(self._get_user_feature(user_id))
                features.append(self._get_topic_feature(topic_id))
            self.val_data.append((user_id, topic_id, features, 1.0))

        for s in self.test_data:
            t = (s[0], s[1])
            if t in positives:
                positives.remove((s[0], s[1]))
            if t in no_interaction:
                no_interaction.remove((s[0], s[1]))

        for s in self.val_data:
            t = (s[0], s[1])
            if t in positives:
                positives.remove((s[0], s[1]))
            if t in no_interaction:
                no_interaction.remove((s[0], s[1]))


        positives = set(interactions)
        negatives = random.sample(list(no_interaction), int(train_negative_frac*len(positives)))

        self.data = []

        for x in positives:
            features = []
            if self.use_features:
                features.append(self._get_user_feature(x[0]))
                features.append(self._get_topic_feature(x[1]))
            self.data.append((x[0], x[1], features, 1.0))

        for x in negatives:
            features = []
            if self.use_features:
                features.append(self._get_user_feature(x[0]))
                features.append(self._get_topic_feature(x[1]))
            self.data.append((x[0], x[1], features, 0.0))

            
    def get_data(self):
        return self.data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data
    

    def _get_user_feature(self, uid):
        if not self.use_features:
            return torch.Tensor()

        if uid not in self.user_features.index:
            return torch.zeros(self.num_user_features).float()

        return torch.tensor(self.user_features.loc[uid, :]).float()

    def _get_topic_feature(self, tid):
        if not self.use_features:
            return torch.Tensor()
        
        if tid not in self.topic_features.index:
            return torch.zeros(self.num_topic_features).float()

        return torch.tensor(self.topic_features.loc[tid, :]).float()

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

    def get_val_dataset(self):
        return LeaveOneOutDS(self.get_val_data(), self.get_user_ids(), self.get_topic_ids())

    def get_test_dataset(self):
        return LeaveOneOutDS(self.get_test_data(), self.get_user_ids(), self.get_topic_ids())
    

class ItemKNNSplitter:
    def __init__(self,
                 df,
                 test_user_frac=0.5,
                 ):
        
        events_df = df[~df['topic_id'].isna()]

        interactions = events_df[['user_id', 'topic_id', 'event_id']].groupby(['user_id', 'topic_id']).count()
        interactions = interactions[interactions['event_id'] >= 5]
        interactions_index = interactions.index

        interactions = interactions.reset_index()
        interactions = interactions.rename(columns={'event_id': 'count'})

        self.matrix = interactions.pivot_table(index='topic_id', columns='user_id', values='count')
        # self.matrix = self.matrix.subtract(self.matrix.mean(axis=1), axis=0)

        self.topics = list(self.matrix.index)

        user_ids = list(set(map(lambda x: x[0], list(interactions_index))))

        test_size = int(test_user_frac * len(user_ids))

        user_ids = random.sample(user_ids, test_size)

        self.test_samples = []

        for uid in user_ids:
            tid = random.choice(self.matrix[~self.matrix[uid].isna()].reset_index()['topic_id'])
            # val = self.matrix[uid][tid]
            self.matrix[uid][tid] = np.nan
            for t in self.topics:
                self.test_samples.append((uid, t, float(t==tid)))


   
    def get_matrix(self):
        return self.matrix

    def get_test_samples(self):
        return self.test_samples