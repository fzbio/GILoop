from gnn import gnn_run
from cnn import cnn_run
from finetune import finetune_run
from tensorflow.keras import backend as K


def train_run(chroms, run_id, seed, source_dataset_name, epoch=50):
    gnn_run(chroms, run_id, seed, source_dataset_name, epoch=epoch)
    K.clear_session()
    cnn_run(chroms, run_id, seed, source_dataset_name, epoch=epoch)
    K.clear_session()
    finetune_run(chroms, run_id, seed, source_dataset_name, epoch=epoch)
    K.clear_session()