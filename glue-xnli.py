# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse

from xlm.utils import bool_flag, initialize_exp
from xlm.evaluation.glue import GLUE
from xlm.evaluation.xnli import XNLI
from xlm.model.embedder import SentenceEmbedder


GLUE_TASKS = ['MNLI-m', 'MNLI-mm', 'QQP', 'QNLI', 'SST-2', 'CoLA', 'MRPC', 'RTE', 'STS-B', 'WNLI', 'AX_MNLI-m']
XNLI_TASKS = ['XNLI']
TASKS = GLUE_TASKS + XNLI_TASKS


# parse parameters
parser = argparse.ArgumentParser(description='Train on GLUE or XNLI')

# main parameters
parser.add_argument("--exp_name", type=str, default="",
                    help="Experiment name")
parser.add_argument("--dump_path", type=str, default="",
                    help="Experiment dump path")
parser.add_argument("--exp_id", type=str, default="",
                    help="Experiment ID")

# evaluation task / pretrained model
parser.add_argument("--transfer_tasks", type=str, default="",
                    help="Transfer tasks, example: 'MNLI-m,RTE,XNLI' ")
parser.add_argument("--model_path", type=str, default="",
                    help="Model location")

# data
parser.add_argument("--data_path", type=str, default="",
                    help="Data path")
parser.add_argument("--max_vocab", type=int, default=-1,
                    help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--min_count", type=int, default=0,
                    help="Minimum vocabulary count")

# batch parameters
parser.add_argument("--max_len", type=int, default=256,
                    help="Maximum length of sentences (after BPE)")
parser.add_argument("--group_by_size", type=bool_flag, default=False,
                    help="Sort sentences by size during the training")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Number of sentences per batch")
parser.add_argument("--max_batch_size", type=int, default=0,
                    help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
parser.add_argument("--tokens_per_batch", type=int, default=-1,
                    help="Number of tokens per batch")

# model / optimization
parser.add_argument("--finetune_layers", type=str, default='0:_1',
                    help="Layers to finetune. 0 = embeddings, _1 = last encoder layer")
parser.add_argument("--weighted_training", type=bool_flag, default=False,
                    help="Use a weighted loss during training")
parser.add_argument("--dropout", type=float, default=0,
                    help="Fine-tuning dropout")
parser.add_argument("--optimizer_e", type=str, default="adam,lr=0.0001",
                    help="Embedder (pretrained model) optimizer")
parser.add_argument("--optimizer_p", type=str, default="adam,lr=0.0001",
                    help="Projection (classifier) optimizer")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="Maximum number of epochs")
parser.add_argument("--epoch_size", type=int, default=-1,
                    help="Epoch size (-1 for full pass over the dataset)")

# debug
parser.add_argument("--debug_train", type=bool_flag, default=False,
                    help="Use valid sets for train sets (faster loading)")
parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                    help="Debug multi-GPU / multi-node within a SLURM job")

# parse parameters
params = parser.parse_args()
if params.tokens_per_batch > -1:
    params.group_by_size = True

# check parameters
assert os.path.isdir(params.data_path)
assert os.path.isfile(params.model_path)

# tasks
params.transfer_tasks = params.transfer_tasks.split(',')
assert len(params.transfer_tasks) > 0
assert all([task in TASKS for task in params.transfer_tasks])

# reload pretrained model
embedder = SentenceEmbedder.reload(params.model_path, params)

# reload langs from pretrained model
params.n_langs = embedder.pretrain_params['n_langs']
params.id2lang = embedder.pretrain_params['id2lang']
params.lang2id = embedder.pretrain_params['lang2id']

# initialize the experiment / build sentence embedder
logger = initialize_exp(params)
scores = {}

# prepare trainers / evaluators
glue = GLUE(embedder, scores, params)
xnli = XNLI(embedder, scores, params)

# run
for task in params.transfer_tasks:
    if task in GLUE_TASKS:
        glue.run(task)
    if task in XNLI_TASKS:
        xnli.run()
