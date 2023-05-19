import torch
from torch import optim, nn
import pytorch_lightning as pl


class GRU4RecNetwork(pl.LightningModule):
    def __init__(self,
                 num_topics,
                 topic_embedding_dim=64,
                 hidden_dim=128,
                 dropout_rate=0.3,
                 loss=nn.Softmax(),
                 ):
        super().__init__()

        self.num_topics = num_topics
        self.topic_embedding_dim = topic_embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.topic_embedding_layer = nn.Embedding(self.num_topics, self.topic_embedding_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.gru_layer = nn.GRU(
            input_size=self.topic_embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        self.linear = nn.Linear(self.hidden_dim, self.num_topics)

        self.activation = nn.LogSoftmax(dim=1)

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
        topic_sequence, topic_sequence_len, label_topic = batch

        log_probas = self(topic_sequence, topic_sequence_len, label_topic)

        loss = self.loss(log_probas, label_topic)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.loss_logs.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        topic_sequence, topic_sequence_len, label_topic = batch

        log_probas = self(topic_sequence, topic_sequence_len, label_topic)

        loss = self.loss(log_probas, label_topic)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        topic_sequence, topic_sequence_len, label_topic = batch

        log_probas = self(topic_sequence, topic_sequence_len, label_topic)
        probas = torch.exp(log_probas)

        self.predict_proba = torch.cat((self.predict_proba, probas.detach().cpu()))


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

