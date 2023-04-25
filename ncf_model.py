import torch
from torch import optim, nn
import lightning.pytorch as pl

class NCFNetwork(pl.LightningModule):
    def __init__(self, num_students, num_topics, student_embedding_dim, topic_embedding_dim, predictive_factors, loss=nn.BCELoss()):
        super().__init__()
        self.student_embedding_layer = nn.Embedding(num_students, student_embedding_dim)
        self.topic_embedding_layer = nn.Embedding(num_topics, topic_embedding_dim)
        
        self.network = nn.Sequential(
            nn.Linear(student_embedding_dim+topic_embedding_dim, predictive_factors*4),
            nn.ReLU(),
            nn.Linear(predictive_factors*4, predictive_factors*2),
            nn.ReLU(),
            nn.Linear(predictive_factors*2, predictive_factors),
            nn.ReLU(),
            nn.Linear(predictive_factors, 1),
            nn.Sigmoid(),
        )

        self.loss = loss

        self.predict_proba = torch.Tensor()

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

        return loss

    def test_step(self, batch, batch_idx):
        student_x, topic_x, y = batch

        y_proba = self(student_x, topic_x)

        self.predict_proba = torch.cat((self.predict_proba, y_proba.detach().cpu()))


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


