import torch
from torch import optim, nn
import pytorch_lightning as pl


class NCFNetwork(pl.LightningModule):
    def __init__(self,
                 num_students,
                 num_topics,
                 predictive_factors=64,
                 loss=nn.BCELoss(),
                 ):
        super().__init__()

        # Parameters are set up to match 3 linear layers configuration
        student_embedding_dim = 2 * predictive_factors
        topic_embedding_dim = 2 * predictive_factors

        self.student_embedding_layer = nn.Embedding(num_students, student_embedding_dim)
        self.topic_embedding_layer = nn.Embedding(num_topics, topic_embedding_dim)

        self.network = nn.Sequential(
            nn.Linear(student_embedding_dim + topic_embedding_dim, 4 * predictive_factors),
            nn.ReLU(),
            nn.Linear(4 * predictive_factors, 2 * predictive_factors),
            nn.ReLU(),
            nn.Linear(2 * predictive_factors, predictive_factors),
            nn.ReLU(),
            nn.Linear(predictive_factors, 1),
            nn.Sigmoid(),
        )

        self.loss = loss

        self.predict_proba = torch.Tensor()
        self.loss_logs = []

        self.save_hyperparameters()

    def forward(self, student_x, topic_x):
        student_emb = self.student_embedding_layer(student_x)
        topic_emb = self.topic_embedding_layer(topic_x)

        x = torch.cat((student_emb, topic_emb), 1)

        proba = self.network(x)

        return proba

    def training_step(self, batch, batch_idx):
        student_x, topic_x, y = batch

        y_proba = self(student_x, topic_x)

        loss = self.loss(y_proba, y)
        self.log("train_loss", loss)

        self.loss_logs.append(loss.item())

        return loss

    def test_step(self, batch, batch_idx):
        student_x, topic_x, y = batch

        y_proba = self(student_x, topic_x)

        self.predict_proba = torch.cat((self.predict_proba, y_proba.detach().cpu()))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    




class FeatureNCFNetwork(pl.LightningModule):
    def __init__(self,
                 num_students,
                 num_topics,
                 num_student_features,
                 num_topic_features,
                 predictive_factors=64,
                 loss=nn.BCELoss(),
                 ):
        super().__init__()

        # Parameters are set up to match 3 linear layers configuration
        student_embedding_dim = 2 * predictive_factors
        topic_embedding_dim = 2 * predictive_factors

        self.student_embedding_layer = nn.Embedding(num_students, student_embedding_dim)
        self.topic_embedding_layer = nn.Embedding(num_topics, topic_embedding_dim)

        self.network = nn.Sequential(
            nn.Linear(student_embedding_dim + topic_embedding_dim + num_student_features + num_topic_features, 4 * predictive_factors),
            nn.ReLU(),
            nn.Linear(4 * predictive_factors, 2 * predictive_factors),
            nn.ReLU(),
            nn.Linear(2 * predictive_factors, predictive_factors),
            nn.ReLU(),
            nn.Linear(predictive_factors, 1),
            nn.Sigmoid(),
        )

        self.loss = loss

        self.predict_proba = torch.Tensor()
        self.loss_logs = []

        self.save_hyperparameters()

    def forward(self, student_ids, topic_ids, student_features, topic_features):
        student_emb = self.student_embedding_layer(student_ids)
        topic_emb = self.topic_embedding_layer(topic_ids)

        x = torch.cat((student_emb, topic_emb, student_features, topic_features), 1)

        proba = self.network(x)

        return proba

    def training_step(self, batch, batch_idx):
        student_ids, topic_ids, student_features, topic_features, y = batch

        y_proba = self(student_ids, topic_ids, student_features, topic_features)

        loss = self.loss(y_proba, y)
        self.log("train_loss", loss)

        self.loss_logs.append(loss.item())

        return loss

    def test_step(self, batch, batch_idx):
        student_ids, topic_ids, student_features, topic_features, y = batch

        y_proba = self(student_ids, topic_ids, student_features, topic_features)


        self.predict_proba = torch.cat((self.predict_proba, y_proba.detach().cpu()))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

