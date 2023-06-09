import torch
from torch import optim, nn
import pytorch_lightning as pl
from loss import BPRLoss

class HAGERecNetwork(pl.LightningModule):
    def __init__(self,
                 num_users,
                 num_topics,
                 num_user_to_topic_relations,
                 num_topic_to_topic_relations,
                 embedding_dim=256,
                 hidden_dim=128,
                 loss=BPRLoss(),
                 ):
        super().__init__()

        self.num_users = num_users
        self.num_topics = num_topics
        self.num_user_to_topic_relations = num_user_to_topic_relations
        self.num_topic_to_topic_relations = num_topic_to_topic_relations

        self.embedding_dim = embedding_dim

        self.hidden_dim = hidden_dim

        self.topic_embedding_layer = nn.Embedding(self.num_topics, self.embedding_dim)
        self.user_embedding_layer = nn.Embedding(self.num_users, self.embedding_dim)
        self.user_to_topic_relations_embedding_layer = nn.Embedding(self.num_user_to_topic_relations, self.embedding_dim)
        self.topic_to_topic_relations_embedding_layer = nn.Embedding(self.num_topic_to_topic_relations,self.embedding_dim)

        self.relu = nn.LeakyReLU()

        self.user_addition_linear_layer = nn.Linear(self.user_embedding_dim, self.hidden_dim)
        self.topic_addition_linear_layer = nn.Linear(self.topic_embedding_dim, self.hidden_dim)
        self.user_multiplication_linear_layer = nn.Linear(self.user_embedding_dim, self.hidden_dim)
        self.topic_multiplication_linear_layer = nn.Linear(self.topic_embedding_dim, self.hidden_dim)

        self.user_attention_layer = nn.Sequential(
            nn.Linear(2*self.embedding_dim, self.hidden_dim),
            self.relu,
            nn.Linear(self.hidden_dim, 1, bias=False)
        )
        self.topic_attention_layer = nn.Sequential(
            nn.Linear(2*self.embedding_dim, self.hidden_dim),
            self.relu,
            nn.Linear(self.hidden_dim, 1, bias=False)
        )

        # TODO: change dim?
        self.softmax = nn.LogSoftmax(dim=1)

        self.loss = loss

        self.predict_proba = torch.Tensor()
        self.loss_logs = []

        self.save_hyperparameters()


    def forward(self, topic_sequence, topic_sequence_len, label_topic):
        topic_sequence_embeddings = self.topic_embedding_layer(topic_sequence)
        topic_sequence_embeddings_dropout = self.dropout(topic_sequence_embeddings)

        packed_topic_sequence = nn.utils.rnn.pack_padded_sequence(topic_sequence_embeddings_dropout, topic_sequence_len.cpu().numpy(), batch_first=True, enforce_sorted=False)
        packed_gru_output, _ = self.gru_layer(packed_topic_sequence)
        unpacked_gru_output, _ = nn.utils.rnn.pad_packed_sequence(packed_gru_output, batch_first=True)

        last_topic_in_sequence_idx = topic_sequence_len - 1

        # dark magic to be able to extract the last sequence value from GRU outputs
        last_topic_in_sequence_idx = last_topic_in_sequence_idx.view(-1, 1, 1).expand(-1, -1, self.hidden_dim)
        gru_output = unpacked_gru_output.gather(dim=1, index=last_topic_in_sequence_idx).squeeze(1)

        linear_output = self.linear(gru_output)

        log_probas = self.activation(linear_output)

        return log_probas


    def training_step(self, batch, batch_idx):
       # user_data, positive_topic_data, negative_topic_data = batch
       #

       # log_probas = self(topic_sequence, topic_sequence_len, label_topic)

       # loss = self.loss(log_probas, label_topic)

       # self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

       # self.loss_logs.append(loss.item())

       # return loss

    def validation_step(self, batch, batch_idx):
        #topic_sequence, topic_sequence_len, label_topic = batch

        #log_probas = self(topic_sequence, topic_sequence_len, label_topic)

        #loss = self.loss(log_probas, label_topic)

        #self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        #topic_sequence, topic_sequence_len, label_topic = batch

        #log_probas = self(topic_sequence, topic_sequence_len, label_topic)
        #probas = torch.exp(log_probas)

        #self.predict_proba = torch.cat((self.predict_proba, probas.detach().cpu()))


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer

    def on_test_epoch_start(self):
        self.predict_proba = torch.Tensor()

