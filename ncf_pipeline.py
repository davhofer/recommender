import pandas as pd

import os
import numpy as np
import torch
import random

SEED = 131

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from data import preprocess_events, LeaveOneOutSplitter, create_topic_features, create_user_features

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl 
import torch
from torch import optim, nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd 
from evaluation import HitRate_NDCG_MRR, metrics_per_topic, HitRate_NDCG_MRR_from_CSV, getMRR, getHitRatio, getNDCG

from ncf_model import NCFNetwork
from data import LeaveOneOutDS


from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar



def create_datasplitter(MATH, GERMAN, USE_FEATURES, users, topics, documents, events, transactions):
    
    
    print("tables loaded.")

    TEST_USER_FRAC = 0.85
    VAL_USER_FRAC = 0.15
    TRAIN_NEGATIVE_FRAC = 2.0

    events_preprocessed = preprocess_events(events, topics, math=MATH, german=GERMAN)
    user_features = None if not USE_FEATURES else create_user_features(users, transactions)
    topic_features = None if not USE_FEATURES else create_topic_features(topics, documents, events)

    NUM_USER_FEATURES = 0
    NUM_TOPIC_FEATURES = 0
    if USE_FEATURES:
        NUM_USER_FEATURES = user_features.shape[1]
        NUM_TOPIC_FEATURES = topic_features.shape[1]

    data_splitter = LeaveOneOutSplitter(
        events_preprocessed,
        device=None,
        use_features=USE_FEATURES,
        user_features=user_features if USE_FEATURES else None,
        topic_features=topic_features if USE_FEATURES else None,
        test_user_frac=TEST_USER_FRAC,
        val_user_frac=VAL_USER_FRAC,
        train_negative_frac=TRAIN_NEGATIVE_FRAC,
        test_sample_strat="newest"
    )
    return data_splitter




def run_model(USE_FEATURES, PREDICTIVE_FACTORS, STUDENT_EMBEDDING_DIM, TOPIC_EMBEDDING_DIM, data_splitter, joint, epochs=10, patience=3, german_ids=[], math_ids=[], batch_size=64):

    train_ds = data_splitter.get_train_dataset()
    val_ds = data_splitter.get_val_dataset()
    test_ds = data_splitter.get_test_dataset()

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    ncf = NCFNetwork(
        num_students=data_splitter.num_students,
        num_topics=data_splitter.num_topics,
        student_embedding_dim=STUDENT_EMBEDDING_DIM,
        topic_embedding_dim=TOPIC_EMBEDDING_DIM,
        predictive_factors=PREDICTIVE_FACTORS,
        use_features=USE_FEATURES,
        intermediate_size_divisor=2,
        output_MLP_num_layers=3,
        num_user_features=data_splitter.num_user_features,
        num_topic_features=data_splitter.num_topic_features,
        loss=nn.BCELoss(),
        joint=joint,
        topic_ids=data_splitter.get_topic_ids(),
        german_ids=german_ids,
        math_ids=math_ids
        )
    

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=patience,
        verbose=True,
        mode="min"
    )

    trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            accumulate_grad_batches=1,
            max_epochs=epochs,
            callbacks=[TQDMProgressBar(refresh_rate=10), early_stop_callback]
    )

    trainer.fit(model=ncf, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=ncf, dataloaders=test_dataloader)

    return ncf.eval_results, ncf