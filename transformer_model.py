import numpy as np
import torch as torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class SequencesDataset(Dataset):
    """
        Sequences Dataset subclass. 
        The dataset is made up of pairs of sequence and next topic ID. The sequences are truncated or padded to 
        have a fixed length.
    """

    def __init__(self, sequences, max_length, pad_id):
        """
        Initialize the SequencesDataset.

        Args:
            sequences (list): A list of sequences, each sequence is a list of integers (topic_ids).
            max_length (int): The maximum length of a sequence. Shorter sequences are padded and longer sequences are truncated.
            pad_id (int): The id to be used for padding shorter sequences.
        """
        # Take the last element as target and the rest as input
        self.X = [torch.tensor(sequence[:-1]) for sequence in sequences if len(sequence) > 1]
        self.Y = [torch.tensor(sequence[-1]) for sequence in sequences if len(sequence) > 1]

        self.pad_id = pad_id
        self.max_length = max_length

    def __len__(self):
        """Returns the number of sequences in the dataset."""
        return len(self.X)

    def __getitem__(self, index):
        """Returns the index-th data item of the dataset."""
        return self.add_padding(self.X[index]), self.Y[index]

    def add_padding(self, sequence):
        """
        Pads or truncates a sequence to have a fixed length equal to self.max_length.

        Args:
            sequence (torch.tensor): The sequence to be padded or truncated.

        Returns:
            torch.tensor: The padded or truncated sequence.
        """
        # Pad the sequence if it's shorter than max_length
        if len(sequence) < self.max_length:
            sequence = [self.pad_id]*(self.max_length - len(sequence)) + list(sequence)
        else:
            sequence = list(sequence)[-self.max_length:]  # truncate the sequence if it's longer than max_length
        return torch.tensor(sequence, dtype=torch.long)


class TransformerModel(nn.Module):
    """
        The model is made up of an embedding layer, a Transformer encoder, and a decoder.
    """
    def __init__(self, n_topics, hidden_size, n_layers, n_heads, dropout):
        """
        Initialize the TransformerModel.

        Args:
            n_topics (int): The size of the vocabulary.
            hidden_size (int): The number of features in the transformer hidden layer.
            n_layers (int): The number of layers in the transformer.
            n_heads (int): The number of heads in the multiheadattention models.
            dropout (float): The dropout value.
        """
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(n_topics, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, n_heads, hidden_size, dropout), n_layers)
        self.decoder = nn.Linear(hidden_size, n_topics)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src):
        """
        Define the forward pass for the transformer model.

        Args:
            src (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the transformer.
        """
        # Embed the input sequence
        src = self.embedding(src)
        
        # Encode the sequence using the transformer encoder
        output = self.transformer_encoder(src)

        # Predict the next topic ID using a linear layer
        output = self.decoder(output[-1])

        # Apply log softmax to the output
        return self.log_softmax(output)


class TransformerRecommender:

    def __init__(self, data, max_length=50, hidden_size=128, num_layers=2, num_heads=2, dropout=0.2):
        """Initializes the TransformerRecommender.

        Args:
            data (DataFrame): Data containing 'user_id' and 'topic_id' columns.
            max_length (int): Maximum sequence length.
            hidden_size (int): Dimension of the model's hidden states.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads in each transformer layer.
            dropout (float): Dropout rate.
        """
        
        new_data = data[['user_id', 'topic_id']].copy()

        # encode the topic IDs
        self.topic_encoder = { 0 : 0 }
        self.topic_decoder = { 0 : 0 }
        for topic_id in new_data['topic_id'].unique():
            if topic_id not in self.topic_encoder:
                self.topic_encoder[topic_id] = len(self.topic_encoder)
                self.topic_decoder[len(self.topic_decoder)] = topic_id

        new_data['topic_id'] = new_data['topic_id'].map(self.topic_encoder)

        # Create a dictionary: {user_id -> sequence of topics for this user}.
        self.sequences = {}
        for user_id, rows in new_data.groupby('user_id'):
            sequence = np.array(rows['topic_id'])
            self.sequences[user_id] = sequence
        
        # Store other parameters and constants.
        self.max_length = max_length
        self.pad_id = 0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_topics = len(self.topic_encoder)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the transformer model and the loss function.
        self.model = TransformerModel(self.num_topics, self.hidden_size, self.num_layers, self.num_heads, self.dropout).to(self.device)
        self.criterion = nn.NLLLoss()

    def _hit_ratio(self, ranklist, gtItem):
        """Computes the Hit Ratio (HR) metric.

        Args:
            ranklist (list): Ranked list of recommended items.
            gtItem (int): Ground truth item.

        Returns:
            int: 1 if the ground truth item is in the ranklist, 0 otherwise.
        """
        return int(gtItem in ranklist)

    def _ndcg(self, ranklist, gtItem):
        """Computes the Normalized Discounted Cumulative Gain (NDCG) metric.

        Args:
            ranklist (list): Ranked list of recommended items.
            gtItem (int): Ground truth item.

        Returns:
            float: NDCG score.
        """
        if gtItem in ranklist:
            return np.log(2) / np.log(ranklist.index(gtItem) + 2)
        else:
            return 0

    def _mrr(self, ranklist, gtItem):
        """Computes the Mean Reciprocal Rank (MRR) metric.

        Args:
            ranklist (list): Ranked list of recommended items.
            gtItem (int): Ground truth item.

        Returns:
            float : MRR score.
        """
        if gtItem in ranklist:
            return 1.0 / (ranklist.index(gtItem) + 1)
        else:
            return 0

    def _split_sequences(self, sequences, window_size):
        """Splits sequences for training and testing.

        Args:
            sequences (dict): All sequences.
            window_size (int): The size of the context window.

        Returns:
            train (list): Training sequences.
            test (list): Testing sequences.
        """
        train, test = [], []
        for user_id, seq in sequences.items():
            if len(seq) > window_size:
                test.append(seq)
                for i in range(2, len(seq) - window_size):
                    train.append(seq[:i])
            else:
                test.append(seq)
        return train, test

    def train(self, num_epochs=2, batch_size=64, window_size=2, lr=0.001, n_samples=100):
        """Trains the recommender system.

        Args:
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            window_size (int): Size of the context window.
            lr (float): Learning rate.

        Returns:
            total_losses (list): List of loss values per epoch.
        """
        train_math_sequences, _ = self._split_sequences(self.sequences, window_size)
        train_dataset = SequencesDataset(train_math_sequences, self.max_length, self.pad_id)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = Adam(self.model.parameters(), lr=lr)

        print('Training the recommender system...')
        total_losses = []
        for epoch in range(num_epochs):
            print(f'-- Epoch {epoch+1}/{num_epochs} --')
            self.model.train()
            epoch_losses = []
            total_loss = 0

            if tqdm is not None:
                progress_bar = tqdm(train_dataloader, desc='Training', total=len(train_dataloader))
            else:
                progress_bar = dataloader

            for i, (X_batch, Y_batch) in enumerate(progress_bar):
                X_batch, Y_batch = X_batch.t().to(self.device), Y_batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(X_batch)

                loss = self.criterion(output, Y_batch)
                loss.backward()

                optimizer.step()
                total_loss += loss.item()

                if i % len(train_dataloader)//n_samples == 0:
                    epoch_losses.append((i, loss.item()))
                
                if tqdm is not None:
                    progress_bar.set_description(f'Training - Loss: {total_loss/(i+1):.4f}')
            
            total_losses.append(epoch_losses)
            print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(train_dataloader)}\n')
        
        return total_losses

    def evaluate(self, batch_size=64, top_k=1, n_samples=100):
        """Evaluates the recommender system.

        Args:
            batch_size (int): Batch size for evaluation.
            top_k (int): Number of top items to consider for evaluation metrics.

        Returns:
            total_losses (list): List of loss values per batch.
        """
        _, test_sequences = self._split_sequences(self.sequences, 1)
        test_dataset = SequencesDataset(test_sequences, self.max_length, self.pad_id)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print('Evaluating the recommender system...')
        self.model.eval()
        total_loss = 0
        total_losses = []
        HR, NDCG, MRR = [], [], []

        if tqdm is not None:
            progress_bar = tqdm(test_dataloader, desc='Evaluation', total=len(test_dataloader))
        else:
            progress_bar = test_dataloader

        with torch.no_grad():
            for i, (X_batch, Y_batch) in enumerate(progress_bar):
                X_batch, Y_batch = X_batch.t().to(self.device), Y_batch.to(self.device)

                output = self.model(X_batch)

                loss = self.criterion(output, Y_batch)
                total_loss += loss.item()

                if i % len(test_dataloader)//n_samples == 0:
                    total_losses.append((i, loss.item()))

                # Compute the top-k items
                _, top_k_items_predicted = torch.topk(output, top_k)
                top_k_items_predicted = top_k_items_predicted.tolist()
                
                # Compute HR, NDCG, and MRR for the batch
                for predicted, true_item in zip(top_k_items_predicted, Y_batch):
                    HR.append(self._hit_ratio(predicted, true_item))
                    NDCG.append(self._ndcg(predicted, true_item))
                    MRR.append(self._mrr(predicted, true_item))

                if tqdm is not None:
                    progress_bar.set_description(f'Evaluation - Loss: {total_loss/(i+1):.4f}')

        # Compute average HR, NDCG, and MRR over the dataset
        HR = np.mean(HR)
        NDCG = np.mean(NDCG)
        MRR = np.mean(MRR)
        avg_loss = total_loss / len(test_dataloader)

        print(f'Evaluation Loss: {avg_loss:.4f}, HR@{top_k}: {HR:.4f}, NDCG@{top_k}: {NDCG:.4f}, MRR@{top_k}: {MRR:.4f}\n')

        return total_losses, HR, NDCG, MRR

    def predict_topk_topics(self, user_id, top_k=5):
        """Predicts top-k topics for a user.

        Args:
            user_id (int): User ID.
            top_k (int): Number of top items to predict.

        Returns:
            top_k_topics (list of tuples): List of top-k topics with their score.
        """
        sequence = self.sequences.get(user_id)
        if sequence is not None:
            # pad sequence if necessary
            if len(sequence) < self.max_length:
                sequence = [self.pad_id]*(self.max_length - len(sequence)) + list(sequence)
            else:
                sequence = list(sequence)[-self.max_length:]  # truncate the sequence if it's longer than max_length
            
            sequence = torch.tensor(sequence, dtype=torch.long).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(sequence)

                top_k = min(top_k, self.num_topics)

                # Compute the top-k items with the scores
                top_k_values, top_k_indices = torch.topk(output, top_k)
                top_k_values = torch.exp(top_k_values).squeeze().tolist()

                # Convert the top-k indices to topic IDs
                top_k_topics = [self.topic_decoder[topic_id] for topic_id in top_k_indices.squeeze().tolist()]

                return list(zip(top_k_topics, top_k_values))

        else:
            return 'User not found'

    def _add_new_topic(self, new_topic):
        """Adds new topic to the recommender system.

        Args:
            new_topic (int): New topic ID to be added.
        """

        if new_topic in self.topic_encoder:
            return 'Topic already exists'

        # Add new topic to the topic encoder
        self.topic_encoder[new_topic] = len(self.topic_encoder)
        self.topic_decoder[len(self.topic_decoder)] = new_topic
        self.num_topics = len(self.topic_encoder)

        self._update_layers()

    def _update_layers(self):
        ## Update embedding layer ##
        # Get the current embedding layer
        old_embedding = self.model.embedding
        old_classes, embedding_size = old_embedding.weight.size()

        # Create a new embedding layer with the increased number of classes
        new_embedding = nn.Embedding(self.num_topics, embedding_size).to(self.device)

        # Copy the weights from the old embedding layer to the new one
        new_embedding.weight.data[:old_classes] = old_embedding.weight.data

        # Replace the old embedding layer with the new one in the model
        self.model.embedding = new_embedding


        ## Update decoder layer ##
        # Get the current decoder layer (linear layer)
        old_output = self.model.decoder
        old_classes, embedding_size = old_output.weight.size()

        # Create a new decoder layer with the increased number of classes
        new_output = nn.Linear(embedding_size, self.num_topics).to(self.device)

        # Copy the weights from the old decoder layer to the new one
        new_output.weight.data[:old_classes, :] = old_output.weight.data
        new_output.bias.data[:old_classes] = old_output.bias.data

        # Replace the old decoder layer with the new one in the model
        self.model.decoder = new_output



    def update_user_sequence(self, user_id, topic_id):
        """Updates the sequence of topics for a user.

        Args:
            user_id (int): User ID.
            topic_id (int): Topic ID to be added to the sequence.
        """

        assert isinstance(topic_id, int) and topic_id > 0
        assert isinstance(user_id, int) and user_id > 0

        
        if topic_id not in self.topic_encoder:
            self._add_new_topic(topic_id)

        if user_id in self.sequences:
            self.sequences[user_id] = np.append(self.sequences[user_id], self.topic_encoder[topic_id])
        else:
            self.sequences[user_id] = np.array([self.topic_encoder[topic_id]])


    def set_user_sequence(self, user_id, sequence):
        """Sets the sequence of topics for a user.

        Args:
            user_id (int): User ID.
            sequence (list): List of topic IDs.
        """
        assert isinstance(user_id, int) and user_id > 0
        assert isinstance(sequence, list)
        assert all(isinstance(topic_id, int) and topic_id > 0 for topic_id in sequence)
        
        new_sequence = []
        for topic_id in sequence:
            if topic_id not in self.topic_encoder:
                self._add_new_topic(topic_id)

            new_sequence.append(self.topic_encoder[topic_id])
        
        self.sequences[user_id] = np.array(new_sequence)



    def get_user_sequence(self, user_id):
        """Returns the sequence of topics for a user.

        Args:
            user_id (int): User ID.

        Returns:
            sequence (list): List of topic IDs.
        """
        assert isinstance(user_id, int) and user_id > 0

        if user_id not in self.sequences:
            return 'User not found'
        
        return [int(self.topic_decoder[topic_id]) for topic_id in self.sequences[user_id]]

    def save_model(self, path):
        """
        Save the model's state_dict to a file.

        Args:
            path (str): Path to the file to save the model to.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Load the model's state_dict from a file.

        Args:
            path (str): Path to the file to load the model from.
        """
        # Load state dictionary
        state_dict = torch.load(path)

        old_num_topics = state_dict['embedding.weight'].size(0)

        # Initialize an old model with the old number of topics
        old_model = TransformerModel(old_num_topics, self.hidden_size, self.num_layers, self.num_heads, self.dropout).to(self.device)

        old_model.load_state_dict(state_dict)

        if old_num_topics > self.num_topics:
            self.model = old_model
            return

        # Copy parameters from the old model to the new model
        self.model.embedding.weight.data[:old_num_topics] = old_model.embedding.weight.data
        self.model.decoder.weight.data[:old_num_topics, :] = old_model.decoder.weight.data
        self.model.decoder.bias.data[:old_num_topics] = old_model.decoder.bias.data