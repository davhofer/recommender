import torch
from torch import optim, nn
import pytorch_lightning as pl


def get_MLP(in_size, out_size, num_layers, softmax_out=False, dimension_decay='linear'):
    assert in_size >= out_size, "in_size must be greater or equal to out_size"
    assert dimension_decay in ['linear', 'exponential'], "dimension_decay must be 'linear' or 'exponential'"
    
    # dimension_decay controls whether the size of hidden layers decreases linearly or exponentially 
    # from in_size to out_size
    # exponential: for each consecutive layer, size is divided by a factor q
    # linear: for each consecutive layer, constant value is subtracted from size

    q = 1
    if dimension_decay == 'exponential':
        q = (in_size/out_size)**(1/num_layers)
        layer_in_float = in_size * q
        layer_out_float = in_size

    layers = []
    for i in range(num_layers):
        if dimension_decay == 'linear':
            layer_in = in_size - int((in_size-out_size) * i/num_layers)
            layer_out = in_size - int((in_size-out_size) * (i+1)/num_layers)
        else:
            layer_in_float = layer_in_float/q
            layer_out_float = layer_out_float/q

            layer_in = round(layer_in_float)
            layer_out = round(layer_out_float)

        layers.append(nn.Linear(layer_in, layer_out))
        layers.append(nn.ReLU())

    # for the final layer, have a single output node with sigmoid activation
    if softmax_out:
        layers.append(nn.Linear(out_size, 1))
        layers.append(nn.Sigmoid())


    return nn.Sequential(*layers)

    


class NCFNetwork(pl.LightningModule):
    def __init__(self,
                 num_students,
                 num_topics,
                 student_embedding_dim=64,
                 topic_embedding_dim=10,
                 predictive_factors=32,
                 use_features=False,
                 num_user_features=0,
                 num_topic_features=0,
                 loss=nn.BCELoss(),
                 ):
        super().__init__()

        self.use_features = use_features
        self.num_user_features = num_user_features
        self.num_topic_features = num_topic_features

        if self.use_features:
            if not (self.num_topic_features > 0 and self.num_user_features > 0):
                print("num_topic_features and num_user_features need to be greater than 0. Init failed!")
                return

        self.student_embedding_layer = nn.Embedding(num_students, student_embedding_dim)
        self.topic_embedding_layer = nn.Embedding(num_topics, topic_embedding_dim)

        # controls how much the size is reduced from the input to the intermediate layer
        first_MLP_dimension_reduction = 2
        first_MLP_layers = 2

        self.user_embed_MLP = get_MLP(student_embedding_dim, student_embedding_dim//first_MLP_dimension_reduction, first_MLP_layers)
        self.user_feature_MLP = get_MLP(num_user_features, num_user_features//first_MLP_dimension_reduction, first_MLP_layers)
        self.topic_embed_MLP = get_MLP(topic_embedding_dim, topic_embedding_dim//first_MLP_dimension_reduction, first_MLP_layers)
        self.topic_feature_MLP = get_MLP(num_topic_features, num_topic_features//first_MLP_dimension_reduction, first_MLP_layers)
        
        intermediate_size = student_embedding_dim//first_MLP_dimension_reduction + topic_embedding_dim//first_MLP_dimension_reduction + self.num_user_features//first_MLP_dimension_reduction + self.num_topic_features//first_MLP_dimension_reduction
        print("intermediate layer size (concatenated):", intermediate_size)

        self.network = get_MLP(intermediate_size, predictive_factors, 3, softmax_out=True, dimension_decay='exponential')

        self.loss = loss

        self.predict_proba = torch.Tensor()
        self.loss_logs = []

        self.save_hyperparameters()


    def forward(self, student_x, topic_x, student_features, topic_features):

        student_emb = self.student_embedding_layer(student_x)
        topic_emb = self.topic_embedding_layer(topic_x)

        student_vec = self.user_embed_MLP(student_emb)
        topic_vec = self.topic_embed_MLP(topic_emb)
        student_f_vec = self.user_feature_MLP(student_features)
        topic_f_vec = self.topic_feature_MLP(topic_features)


        x = torch.cat((student_vec, topic_vec, student_f_vec, topic_f_vec), 1)

        proba = self.network(x)

        return proba


    def training_step(self, batch, batch_idx):
        student_x, topic_x, student_features, topic_features, y = batch

        y_proba = self(student_x, topic_x, student_features, topic_features)

        loss = self.loss(y_proba, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.loss_logs.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        student_x, topic_x, student_features, topic_features, y = batch

        y_proba = self(student_x, topic_x, student_features, topic_features)

        loss = self.loss(y_proba, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        self.predict_proba = torch.Tensor()

    def test_step(self, batch, batch_idx):
        student_x, topic_x, student_features, topic_features, y = batch

        y_proba = self(student_x, topic_x, student_features, topic_features)

        self.predict_proba = torch.cat((self.predict_proba, y_proba.detach().cpu()))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
