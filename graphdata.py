import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
from data import train_test_val_split


def filter_topic_trees(events_preprocessed, topic_trees, include_leafs_and_roots=False):
    math_topics = set(events_preprocessed[events_preprocessed['is_math'] == 1]['topic_id'])
    german_topics = set(events_preprocessed[events_preprocessed['is_math'] == 0]['topic_id'])
    math_topic_trees = topic_trees[
        (topic_trees['parent_id'].isin(math_topics)) | (topic_trees['child_id'].isin(math_topics))].copy()
    german_topic_trees = topic_trees[
        (topic_trees['parent_id'].isin(german_topics)) | (topic_trees['child_id'].isin(german_topics))].copy()

    math_topic_trees['is_math'] = 1
    german_topic_trees['is_math'] = 0

    filtered_topic_trees = pd.concat((math_topic_trees, german_topic_trees))

    if not include_leafs_and_roots:
        filtered_topic_trees = filtered_topic_trees[~filtered_topic_trees['parent_id'].isna()]
        filtered_topic_trees = filtered_topic_trees[~filtered_topic_trees['child_id'].isna()]
        filtered_topic_trees['parent_id'] = filtered_topic_trees['parent_id'].astype(int)

    return filtered_topic_trees


@dataclass
class EntitiesGraph:
    user_topic_relations: List[Tuple[int, int]]
    user_to_topics_and_relations: Dict[int, torch.Tensor]  # num_neighbours x 2
    topic_to_users_and_relations: Dict[int, torch.Tensor]  # num_neighbours x 2
    topic_to_topics_and_relations: Dict[int, torch.Tensor]  # num_neighbours x 2
    num_topic_to_topic_relations: int


class GraphDS(Dataset):
    def __init__(self, data: List[Tuple[int, int, int]], knowledge_graph: EntitiesGraph):
        self.data = data
        self.knowledge_graph = knowledge_graph

    def get_user_data(self, user_id):
        neighbour_users_and_relations = self.knowledge_graph.user_to_topics_and_relations.get(user_id, torch.empty((0, 2)))
        return torch.tensor(user_id), neighbour_users_and_relations

    def get_topic_data(self, topic_id):
        neighbour_users_and_relations = self.knowledge_graph.topic_to_users_and_relations.get(topic_id, torch.empty((0, 2)))
        neighbour_topics_and_relations = self.knowledge_graph.topic_to_topics_and_relations.get(topic_id, torch.empty((0, 2)))
        return torch.tensor(topic_id), neighbour_users_and_relations, neighbour_topics_and_relations


class LeaveOneOutGraphDS(GraphDS):
    def __init__(self, data: List[Tuple[int, int, int]], knowledge_graph: EntitiesGraph):
        super().__init__(data, knowledge_graph)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_id, topic_id, label = self.data[index]
        return self.get_user_data(user_id), self.get_topic_data(topic_id), torch.Tensor([label])


class PositiveNegativeGraphDS(GraphDS):
    def __init__(self, data: List[Tuple[int, int, int]], knowledge_graph: EntitiesGraph):
        super().__init__(data, knowledge_graph)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_id, positive_topic_id, negative_topic_id = self.data[index]
        return self.get_user_data(user_id), self.get_topic_data(positive_topic_id), self.get_topic_data(negative_topic_id)


def construct_knowledge_graph(
        user_index: Dict[int, int],
        topic_index: Dict[int, int],
        positive_examples_by_user: Dict[int, List[int]],
        topic_trees: pd.DataFrame,
) -> EntitiesGraph:
    user_topic_relations = []
    user_to_topics_and_relations: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    topic_to_users_and_relations: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    for user, topics in positive_examples_by_user.items():
        user_idx = user_index[user]
        for topic in topics:
            topic_idx = topic_index[topic]
            relation_idx = len(user_topic_relations)
            user_topic_relations.append((user_idx, topic_idx))
            user_to_topics_and_relations[user_idx].append((topic_idx, relation_idx))
            topic_to_users_and_relations[topic_idx].append((user_idx, relation_idx))

    topic_to_topics_and_relations: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for relation_idx, topic_from, topic_to in topic_trees[['topic_id', 'parent_id', 'child_id']].values.tolist():
        topic_from_idx = topic_index[topic_from]
        topic_to_idx = topic_index[topic_to]
        topic_to_topics_and_relations[topic_from_idx].append((topic_to_idx, relation_idx))
        topic_to_topics_and_relations[topic_to_idx].append((topic_from_idx, relation_idx))

    num_topic_to_topic_relations = topic_trees['topic_id'].nunique()

    def to_tensor(user_data: Dict[int, List[Tuple[int, int]]]) -> Dict[int, torch.Tensor]:
        return {u: torch.tensor(data) for u, data in user_data.items()}

    return EntitiesGraph(
        user_topic_relations=user_topic_relations,
        user_to_topics_and_relations=to_tensor(user_to_topics_and_relations),
        topic_to_users_and_relations=to_tensor(topic_to_users_and_relations),
        topic_to_topics_and_relations=to_tensor(topic_to_topics_and_relations),
        num_topic_to_topic_relations=num_topic_to_topic_relations,
    )


def get_user_topic_index(users, topics):
    user_index = {}
    for user_idx, user in enumerate(users):
        user_index[user] = user_idx
    topic_index = {}
    for topic_idx, topic in enumerate(topics):
        topic_index[topic] = topic_idx

    return user_index, topic_index


def remap_train_data(user_index, topic_index, train_data, val_data, test_data):
    remapped_train_data = []
    for user, positive_topic, negative_topic in train_data:
        remapped_train_data.append((user_index[user], topic_index[positive_topic], topic_index[negative_topic]))

    remapped_val_data = []
    for user, topic, label in val_data:
        remapped_val_data.append((user_index[user], topic_index[topic], label))

    remapped_test_data = []
    for user, topic, label in test_data:
        remapped_test_data.append((user_index[user], topic_index[topic], label))

    return remapped_train_data, remapped_val_data, remapped_test_data


class LeaveOneOutGraphSplitter:
    def __init__(self,
                 preprocessed_df,
                 topic_trees_df,
                 test_user_frac=0.5,
                 val_user_frac=0.5,
                 train_negative_frac=1.0,
                 test_sample_strat="newest",
                 ):

        self.df = preprocessed_df
        self.topic_trees = filter_topic_trees(
            events_preprocessed=self.df,
            topic_trees=topic_trees_df,
            include_leafs_and_roots=False,
        )

        self.test_user_frac = test_user_frac
        self.val_user_frac = val_user_frac

        self.train_negative_frac = train_negative_frac
        self.test_sample_strat = test_sample_strat

        self.user_ids = list(self.df['user_id'].unique())
        self.interaction_matrix_topic_ids = set(self.df['topic_id'].unique())
        self.knowledge_graph_topic_ids = set(self.topic_trees['parent_id'].unique()).union(
                                             self.topic_trees['child_id'].unique()
        )
        self.topic_ids = list(self.interaction_matrix_topic_ids.union(self.knowledge_graph_topic_ids))

        self.num_students = len(self.user_ids)
        self.num_topics = len(self.topic_ids)

        if test_user_frac + val_user_frac > 1:
            raise ValueError("'test_user_frac' + 'val_user_frac' must be <= 1.0")

        self.data, self.val_data, self.test_data, self.train_positive_examples_by_user = train_test_val_split(
            df=self.df,
            test_user_frac=self.test_user_frac,
            val_user_frac=self.val_user_frac,
            train_negative_frac=self.train_negative_frac,
            test_sample_strat=self.test_sample_strat,
            match_users_in_train_negative_samples=True,
            return_positive_negative_pairs_in_train=True,
            return_train_positive_examples=True,
        )

        self.user_index, self.topic_index = get_user_topic_index(users=self.user_ids, topics=self.topic_ids)

        self.knowledge_graph = construct_knowledge_graph(
            user_index=self.user_index,
            topic_index=self.topic_index,
            positive_examples_by_user=self.train_positive_examples_by_user,
            topic_trees=self.topic_trees,
        )

        self.data, self.val_data, self.test_data = remap_train_data(
            user_index=self.user_index,
            topic_index=self.topic_index,
            train_data=self.data,
            val_data=self.val_data,
            test_data=self.test_data,
        )

    def get_num_users(self):
        return len(self.user_ids)

    def get_num_topics(self):
        return len(self.topic_ids)

    def get_num_user_to_topic_relations(self):
        return len(self.knowledge_graph.user_topic_relations)

    def get_num_topic_to_topic_relations(self):
        return self.knowledge_graph.num_topic_to_topic_relations

    def get_train_dataset(self):
        return PositiveNegativeGraphDS(data=self.data, knowledge_graph=self.knowledge_graph)

    def get_val_dataset(self):
        return LeaveOneOutGraphDS(data=self.val_data, knowledge_graph=self.knowledge_graph)

    def get_test_dataset(self):
        return LeaveOneOutGraphDS(data=self.test_data, knowledge_graph=self.knowledge_graph)
