import torch
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import numpy as np


PAD_TOPIC_ID = 0


def preprocess_events(df, topics, math=True, german=True):
    assert math or german, "Either math or german must be set to True."

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


    if not (math and german):
        if math:
            df = df[df['is_math']==1]
        elif german:
            df = df[df['is_math']==0]

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
        user, topic, user_f, topic_f, y = self.data[index]

        user = self.user_ids.index(user)
        user = torch.tensor(user)

        topic = self.topic_ids.index(topic)
        topic = torch.tensor(topic)

        y = torch.Tensor([y])
        return user, topic, user_f, topic_f, y

# TODO: test how many negative examples in test/val set occur in training set

def train_test_val_split(df, test_user_frac, val_user_frac, train_negative_frac, test_sample_strat="newest", verbose=1):
    if test_user_frac + val_user_frac > 1:
            print("'test_user_frac' + 'val_user_frac' must be <= 1.0")
            return
    if test_sample_strat not in ['newest', 'random']:
            print("'test_sample_strat' should either be 'newest' or 'random'!")
            return
    

    user_ids = list(df['user_id'].unique())
    topic_ids = list(df['topic_id'].unique())

    interactions = list(df.groupby(['user_id', 'topic_id']).count().index)

    all_pairings = {(user, topic) for user in user_ids for topic in topic_ids}

    positives = set(interactions)
    non_interactions = all_pairings - positives


    test_size = int(test_user_frac * len(user_ids))
    val_size = int(val_user_frac * len(user_ids))



    val_test_interactions = []
    if test_sample_strat == 'random':
        for id in user_ids:
            user_interactions = list(filter(lambda x: x[0] == id, interactions))
            s = random.choice(user_interactions)
            val_test_interactions.append(s)
    else:
        user_last_event = df[['user_id', 'event_date']].groupby('user_id').max()
        df['test_val_set'] = df.apply(lambda row: row['event_date'] == user_last_event['event_date'][row['user_id']], axis=1)
        val_test_interactions = list(df[df['test_val_set']].groupby(['user_id', 'topic_id']).count().index)

    if verbose:
        print("Sampled initial validation and test interactions")
    
    # get test interactions
    test_samples = random.sample(val_test_interactions, test_size)
    # get validation interactions. make sure there is no overlap
    val_samples = random.sample(list(set(val_test_interactions) - set(test_samples)), val_size)

    # add other user-topic pairs that are not in the interactions set in order to rank the test sample
    test_data = []
    for user_id, topic_id in test_samples:
        for t in topic_ids:
            # don't add a user_topic pair if it is an interaction
            if t != topic_id and (user_id, t) in interactions:
                continue
            label = 1.0 if topic_id == t else 0.0
            test_data.append((user_id, t, label))

    if verbose:
        print("Completed test dataset")

    # add other user-topic pairs that are not in the interactions set in order to rank the test sample
    val_data = []
    for user_id, topic_id in val_samples:
        for t in topic_ids:
            # don't add a user_topic pair if it is an interaction
            if t != topic_id and (user_id, t) in interactions:
                continue

            label = 1.0 if topic_id == t else 0.0
            val_data.append((user_id, t, label))

    if verbose:
        print("Completed validation dataset")

    positives = set(interactions) - set(val_test_interactions)

    #####################################################################################
    # TODO: do we include the negative samples from the training set in the test/val set?
    #####################################################################################

    negatives = random.sample(list(non_interactions), int(train_negative_frac*len(positives)))

    train_data = []

    for x in positives:
        train_data.append((x[0], x[1], 1.0))

    for x in negatives:
        train_data.append((x[0], x[1], 0.0))

    print("Completed train dataset")

    return train_data, val_data, test_data

    



"""
Testing Procedure:
test set consists of a single interaction for each user (or a subset of users)
that was removed from the training set.
"""

class LeaveOneOutSplitter:
    def __init__(self,
                 preprocessed_df,
                 use_features=False,
                 user_features=None,
                 topic_features=None,
                 test_user_frac=0.5,
                 val_user_frac=0.5,
                 train_negative_frac=1.0,
                 test_sample_strat="newest",
                 verbose=1
                 ):
        if test_sample_strat not in ['newest', 'random']:
            print("'test_sample_strat' should either be 'newest' or 'random'!")
            return
        
        """
        TODO: adapt train/test split
        - no negative samples in test and val set
        - instead, just ignore interactions which appear in train & val set, i.e. don't make predictions for those interactions, eliminate them from the ranking

        global interactions, non-interactions
        train set: pos = subset of all interactions, neg = subset of all non-interactions
        val set: pos = disjoint subset of all interactions, neg = all non-interactions for those interactions users
        """

        self.df = preprocessed_df 
        self.train_negative_frac = train_negative_frac
        self.test_sample_strat = test_sample_strat

        self.use_features = use_features
        self.user_features = user_features
        self.topic_features = topic_features

        self.user_ids = list(self.df['user_id'].unique())
        self.topic_ids = list(self.df['topic_id'].unique())

        self.num_students = len(self.user_ids)
        self.num_topics = len(self.topic_ids)


        if test_user_frac + val_user_frac > 1:
            print("'test_user_frac' + 'val_user_frac' must be <= 1.0")
            return 

        if self.use_features:
            if self.user_features is None or self.topic_features is None:
                print("No features have been passed but use_features is True. Init failed!")
                return 
            
            self.num_user_features = self.user_features.shape[1]

            self.num_topic_features = self.topic_features.shape[1]
        
        else:
            self.num_user_features = 0
            self.num_topic_features = 0


        self.data, self.val_data, self.test_data = train_test_val_split(preprocessed_df, test_user_frac, val_user_frac, train_negative_frac, test_sample_strat, verbose=verbose)

        if verbose:
            print("Adding features...")

        self.change_features(self.use_features, self.user_features, self.topic_features)


    def change_features(self, use_features, user_features, topic_features):
        self.use_features = use_features

        self.user_features = user_features 
        self.topic_features = topic_features


        self.data = list(map(lambda x: (x[0], x[1], self._get_user_feature(x[0]), self._get_topic_feature(x[1]), x[-1]), self.data))
        
        self.val_data = list(map(lambda x: (x[0], x[1], self._get_user_feature(x[0]), self._get_topic_feature(x[1]), x[-1]), self.val_data))

        self.test_data = list(map(lambda x: (x[0], x[1], self._get_user_feature(x[0]), self._get_topic_feature(x[1]), x[-1]), self.test_data))

        

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
                 test_user_frac=0.5
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

        # NOTE: we don't train anything here so we don't need a validation set

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


class SequentialSplitter:
    def __init__(self, preprocessed_df):
        self.df = preprocessed_df

        self.topic_ids = list(self.df['topic_id'].unique())
        self.num_topics = len(self.topic_ids)

        # grouping sorted by timestep topic interactions by user
        self.df = self.df.sort_values(by='event_date')[['user_id', 'topic_id']].groupby('user_id').agg(
            list).reset_index()
        self.df = self.df.rename(columns={'topic_id': 'initial_topic_seq'})

        # using last 3 topic ids in the sequence [... t1, t2, t3]: t1 -> for training, t2 for validation, t3 for testing
        self.df['train_topic_id'] = self.df['initial_topic_seq'].apply(lambda x: x[-3])
        self.df['val_topic_id'] = self.df['initial_topic_seq'].apply(lambda x: x[-2])
        self.df['test_topic_id'] = self.df['initial_topic_seq'].apply(lambda x: x[-1])

        self.df['train_topic_seq'] = self.df['initial_topic_seq'].apply(lambda x: x[:-3])
        self.df['val_topic_seq'] = self.df['initial_topic_seq'].apply(lambda x: x[:-2])
        self.df['test_topic_seq'] = self.df['initial_topic_seq'].apply(lambda x: x[:-1])

        self.df['train_topic_seq_len'] = self.df['train_topic_seq'].apply(lambda x: len(x))
        self.df['val_topic_seq_len'] = self.df['val_topic_seq'].apply(lambda x: len(x))
        self.df['test_topic_seq_len'] = self.df['test_topic_seq'].apply(lambda x: len(x))

        # padding the topic sequences
        max_seq_len_train = self.df['train_topic_seq'].apply(lambda x: len(x)).max()
        max_seq_len_val = max_seq_len_train + 1
        max_seq_len_test = max_seq_len_train + 2

        self.df['train_topic_seq'] = self.df['train_topic_seq'].apply(
            lambda x: x + [PAD_TOPIC_ID] * (max_seq_len_train - len(x)))
        self.df['val_topic_seq'] = self.df['val_topic_seq'].apply(
            lambda x: x + [PAD_TOPIC_ID] * (max_seq_len_val - len(x)))
        self.df['test_topic_seq'] = self.df['test_topic_seq'].apply(
            lambda x: x + [PAD_TOPIC_ID] * (max_seq_len_test - len(x)))

        # extracting the data to lists
        train_topic_seq = list(self.df['train_topic_seq'])
        val_topic_seq = list(self.df['val_topic_seq'])
        test_topic_seq = list(self.df['test_topic_seq'])

        train_topic_seq_len = self.df['train_topic_seq_len'].tolist()
        val_topic_seq_len = self.df['val_topic_seq_len'].tolist()
        test_topic_seq_len = self.df['test_topic_seq_len'].tolist()

        train_label = self.df['train_topic_id'].tolist()
        val_label = self.df['val_topic_id'].tolist()
        test_label = self.df['test_topic_id'].tolist()

        self.data = list(zip(train_topic_seq, train_topic_seq_len, train_label))
        self.val_data = list(zip(val_topic_seq, val_topic_seq_len, val_label))
        self.test_data = list(zip(test_topic_seq, test_topic_seq_len, test_label))

    def get_num_topics(self):
        return self.num_topics

    def get_topic_ids(self):
        return self.topic_ids

    def get_data(self):
        return self.data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data

    def get_train_dataset(self):
        return SequentialDS(self.get_data(), self.get_topic_ids())

    def get_val_dataset(self):
        return SequentialDS(self.get_val_data(), self.get_topic_ids())

    def get_test_dataset(self):
        return SequentialDS(self.get_test_data(), self.get_topic_ids())


class SequentialDS(Dataset):
    def __init__(self, data, topic_ids):
        self.data = data
        self.topic_ids = topic_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        topic_sequence, sequence_len, label_topic = self.data[index]

        topic_sequence = [self.topic_ids.index(t) for t in topic_sequence]

        label_topic = self.topic_ids.index(label_topic)
        label_topic = torch.tensor([label_topic])

        return torch.tensor(topic_sequence), torch.tensor([sequence_len]), label_topic
