# XLM

PyTorch original implementation of [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291).  
Provides a cross-lingual implementation of BERT, with state-of-the-art results on XNLI, and unsupervised MT.
Provides a monolingual implementation of BERT, with better performance on the GLUE benchmark.

Model | Score | CoLA | SST2 | MRPC | STS-B | QQP | MNLI_m | MNLI_mm | QNLI | RTE | WNLI | AX
|:---: |:---: |:---: | :---: |:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
`BERT` | 80.5 | 60.5 | 94.9 | 89.3/85.4 | 87.6/86.5 | 72.1/89.3 | 86.7 | 85.9 | 92.7 | 70.1 | 65.1 | 39.6
`XLM_en` | **82.8** | **62.9** | **95.6** | **90.7/87.1** | **88.8/88.2** | **73.2/89.8** | **89.1** | **88.5** | **94.0** | **76.0** | **71.9** | **44.7**

<br>
<br>

![Model](https://dl.fbaipublicfiles.com/XLM/xlm_figure.jpg)

<br>
<br>

XLM contains code for:
- Language model pretraining:
    - Causal Language Model (CLM) - monolingual
    - Masked Language Model (MLM) - monolingual
    - Translation Language Model (TLM) - cross-lingual
- Supervised / Unsupervised MT training:
    - Denoising auto-encoder
    - Parallel data training
    - Online back-translation
- XNLI fine-tuning
- GLUE fine-tuning

XLM supports multi-GPU and multi-node training.


## Pretrained models

We provide our pretrained English model and cross-lingual language models, all trained with the MLM objective (see training command below):

| Languages        | Pretraining | Model                                                               | BPE codes                                                     | Vocabulary                                                     |
| ---------------- | ----------- |:-------------------------------------------------------------------:|:-------------------------------------------------------------:| --------------------------------------------------------------:|
| English          |     MLM     | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_en_2048.pth)         | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_en)      | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_en)    |
| English-French   |     MLM     | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_enfr_1024.pth)       | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_enfr)    | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_enfr)    |
| English-German   |     MLM     | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_ende_1024.pth)       | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_ende)    | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_ende)    |
| English-Romanian |     MLM     | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_enro_1024.pth)       | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_enro)    | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_enro)    |
| XNLI-15          |     MLM     | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_xnli15_1024.pth)     | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_xnli_15) | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15) |
| XNLI-15          |  MLM + TLM  | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth) | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_xnli_15) | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15) |


Our **XLM** PyTorch English model is trained on the same data than the pretrained **BERT** [TensorFlow](https://github.com/google-research/bert) model (Wikipedia + Toronto Book Corpus). Our implementation does not use the next-sentence prediction task and has only 12 layers but higher capacity (665M parameters). Overall, our model achieves a better performance than the original BERT on all GLUE tasks (cf. table above for comparison).

The English-French, English-German and English-Romanian models are the ones we used in the paper for MT pretraining. They are trained with monolingual data only, with the MLM objective. If you use these models, you should use the same data preprocessing / BPE codes to preprocess your data. See the preprocessing commands in [get-data-nmt.sh](https://github.com/facebookresearch/XLM/blob/master/get-data-nmt.sh).

XNLI-15 is the model used in the paper for XNLI fine-tuning. It handles English, French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili and Urdu. It is trained with the MLM and the TLM objectives. For this model we used a different preprocessing than for the MT models (such as lowercasing and accents removal).

## Generating cross-lingual sentence representations

This [notebook](generate-embeddings.ipynb) provides an example to quickly obtain sentence representations from a pretrained model.

## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (currently tested on version 0.4 and 1.0)
- [fastBPE](https://github.com/glample/fastBPE) (generate and apply BPE codes)
- [Moses](http://www.statmt.org/moses/) (scripts to clean and tokenize text only - no installation required)
- [Apex](https://www.github.com/nvidia/apex) (for fp16 training)


## Supervised / Unsupervised MT experiments

### Download / preprocess data

To download the data required for the unsupervised MT experiments, simply run:

```
git clone https://github.com/facebookresearch/XLM.git
cd XLM
```

And one of the three commands below:

```
./get-data-nmt.sh --src en --tgt fr
./get-data-nmt.sh --src de --tgt en
./get-data-nmt.sh --src en --tgt ro
```

for English-French, German-English, or English-Romanian experiments. The script will successively:
- download Moses scripts, download and compile fastBPE
- download, extract, tokenize, apply BPE to monolingual and parallel test data
- binarize all datasets

If you want to use our pretrained models, you need to have an exactly identical vocabulary. Since small differences can happen during preprocessing, we recommend that you use our BPE codes and vocabulary (although you should get something almost identical if you learn the codes and compute the vocabulary yourself). This will ensure that the vocabulary of your preprocessed data perfectly matches the one of our pretrained models, and that there is not a word / index mismatch. To do so, simply run:

```
wget https://dl.fbaipublicfiles.com/XLM/codes_enfr
wget https://dl.fbaipublicfiles.com/XLM/vocab_enfr

./get-data-nmt.sh --src en --tgt fr --reload_codes codes_enfr --reload_vocab vocab_enfr
```

`get-data-nmt.sh` contains a few parameters defined at the beginning of the file:
- `N_MONO` number of monolingual sentences for each language (default 5000000)
- `CODES` number of BPE codes (default 60000)
- `N_THREADS` number of threads in data preprocessing (default 16)

The default number of monolingual data is 5M sentences, but using more monolingual data will significantly improve the quality of pretrained models. In practice, the models we release for MT are trained on all NewsCrawl data available, i.e. about 260M, 200M and 65M sentences for German, English and French respectively.

The script should output a data summary that contains the location of all files required to start experiments:

```
===== Data summary
Monolingual training data:
    en: ./data/processed/en-fr/train.en.pth
    fr: ./data/processed/en-fr/train.fr.pth
Monolingual validation data:
    en: ./data/processed/en-fr/valid.en.pth
    fr: ./data/processed/en-fr/valid.fr.pth
Monolingual test data:
    en: ./data/processed/en-fr/test.en.pth
    fr: ./data/processed/en-fr/test.fr.pth
Parallel validation data:
    en: ./data/processed/en-fr/valid.en-fr.en.pth
    fr: ./data/processed/en-fr/valid.en-fr.fr.pth
Parallel test data:
    en: ./data/processed/en-fr/test.en-fr.en.pth
    fr: ./data/processed/en-fr/test.en-fr.fr.pth
```

### Pretrain a language model (with MLM)

The following script will pretrain a model with the MLM objective for English and French:

```
python train.py

## main parameters
--exp_name test_enfr_mlm                # experiment name
--dump_path ./dumped/                   # where to store the experiment

## data location / training objective
--data_path ./data/processed/en-fr/     # data location
--lgs 'en-fr'                           # considered languages
--clm_steps ''                          # CLM objective
--mlm_steps 'en,fr'                     # MLM objective

## transformer parameters
--emb_dim 1024                          # embeddings / model dimension
--n_layers 6                            # number of layers
--n_heads 8                             # number of heads
--dropout 0.1                           # dropout
--attention_dropout 0.1                 # attention dropout
--gelu_activation true                  # GELU instead of ReLU

## optimization
--batch_size 32                         # sequences per batch
--bptt 256                              # sequences length
--optimizer adam,lr=0.0001              # optimizer
--epoch_size 200000                     # number of sentences per epoch
--validation_metrics _valid_mlm_ppl     # validation metric (when to save the best model)
--stopping_criterion _valid_mlm_ppl,10  # end experiment if stopping criterion does not improve
```

If parallel data is available, the TLM objective can be used with `--mlm_steps 'en-fr'`. To train with both the MLM and TLM objective, you can use `--mlm_steps 'en,fr,en-fr'`. We provide models trained with the MLM objective for English-French, English-German and English-Romanian, along with the BPE codes and vocabulary used to preprocess the data.

### Train on unsupervised MT from a pretrained model

You can now use the pretrained model for Machine Translation. To download a model trained with the command above on the MLM objective, and the corresponding BPE codes, run:

```
wget -c https://dl.fbaipublicfiles.com/XLM/mlm_enfr_1024.pth
```

If you preprocessed your dataset in `./data/processed/en-fr/` with the provided BPE codes `codes_enfr` and vocabulary `vocab_enfr`, you can pretrain your NMT model with `mlm_enfr_1024.pth` and run:

```
python train.py

## main parameters
--exp_name unsupMT_enfr                                       # experiment name
--dump_path ./dumped/                                         # where to store the experiment
--reload_model 'mlm_enfr_1024.pth,mlm_enfr_1024.pth'          # model to reload for encoder,decoder

## data location / training objective
--data_path ./data/processed/en-fr/                           # data location
--lgs 'en-fr'                                                 # considered languages
--ae_steps 'en,fr'                                            # denoising auto-encoder training steps
--bt_steps 'en-fr-en,fr-en-fr'                                # back-translation steps
--word_shuffle 3                                              # noise for auto-encoding loss
--word_dropout 0.1                                            # noise for auto-encoding loss
--word_blank 0.1                                              # noise for auto-encoding loss
--lambda_ae '0:1,100000:0.1,300000:0'                         # scheduling on the auto-encoding coefficient

## transformer parameters
--encoder_only false                                          # use a decoder for MT
--emb_dim 1024                                                # embeddings / model dimension
--n_layers 6                                                  # number of layers
--n_heads 8                                                   # number of heads
--dropout 0.1                                                 # dropout
--attention_dropout 0.1                                       # attention dropout
--gelu_activation true                                        # GELU instead of ReLU

## optimization
--tokens_per_batch 2000                                       # use batches with a fixed number of words
--batch_size 32                                               # batch size (for back-translation)
--bptt 256                                                    # sequence length
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001  # optimizer
--epoch_size 200000                                           # number of sentences per epoch
--eval_bleu true                                              # also evaluate the BLEU score
--stopping_criterion 'valid_en-fr_mt_bleu,10'                 # validation metric (when to save the best model)
--validation_metrics 'valid_en-fr_mt_bleu'                    # end experiment if stopping criterion does not improve
```

The parameters of your Transformer model have to be identical to the ones used for pretraining (or you will have to slightly modify the code to only reload existing parameters). After 8 epochs on 8 GPUs, the above command should give you something like this:

```
epoch               ->     7
valid_fr-en_mt_bleu -> 28.36
valid_en-fr_mt_bleu -> 30.50
test_fr-en_mt_bleu  -> 34.02
test_en-fr_mt_bleu  -> 36.62
```

## Cross-lingual text classification (XNLI)

XLMs can be used to build cross-lingual classifiers. After fine-tuning an XLM model on an English training corpus for instance (e.g. of sentiment analysis, natural language inference), the model is still able to make accurate predictions at test time in other languages, for which there is very little or no training data. This approach is usually referred to as "zero-shot cross-lingual classification".

### Get the right tokenizers

Before running the scripts below, make sure you download the tokenizers from the [tools/](https://github.com/facebookresearch/XLM/tree/master/tools) directory.

### Download / preprocess monolingual data

This script will download and preprocess the Wikipedia datasets in the 15 languages that are part of XNLI:

```
for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do
  ./get-data-wiki.sh $lg
done
```

Downloading the Wikipedia dumps make take several hours. The *get-data-wiki.sh* script will automatically download Wikipedia dumps, extract raw sentences, clean and tokenize them, apply BPE codes and binarize the data. Note that in our experiments we also concatenated the [Toronto Book Corpus](http://yknzhu.wixsite.com/mbweb) to the English Wikipedia.

For Chinese and Thai you will need a special tokenizer that you can install using the commands below. For all other languages, the data will be tokenized with Moses scripts.

```
# Thai - https://github.com/PyThaiNLP/pythainlp
pip install pythainlp

# Chinese
cd tools/
wget https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip
unzip stanford-segmenter-2018-10-16.zip
```

### Download / preprocess parallel data

This script will download and preprocess parallel data that can be used for the TLM objective:

```
lg_pairs="ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh"
for lg_pair in $lg_pairs; do
  ./get-data-para.sh $lg_pair
done
```

### Download / preprocess XNLI data

This script will download and preprocess the XNLI corpus:

```
./get-data-xnli.sh
```

### Pretrain a language model (with MLM and TLM)

The following script will pretrain a model with the MLM and TLM objectives for the 15 XNLI languages:

```
python train.py

## main parameters
--exp_name train_xnli_mlm_tlm            # experiment name
--dump_path ./dumped/                    # where to store the experiment

## data location / training objective
--data_path ./data/processed/XLM15/                   # data location
--lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh'  # considered languages
--clm_steps ''                                        # CLM objective
--mlm_steps 'ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh,en-ar,en-bg,en-de,en-el,en-es,en-fr,en-hi,en-ru,en-sw,en-th,en-tr,en-ur,en-vi,en-zh,ar-en,bg-en,de-en,el-en,es-en,fr-en,hi-en,ru-en,sw-en,th-en,tr-en,ur-en,vi-en,zh-en'  # MLM objective

## transformer parameters
--emb_dim 1024                           # embeddings / model dimension
--n_layers 12                            # number of layers
--n_heads 8                              # number of heads
--dropout 0.1                            # dropout
--attention_dropout 0.1                  # attention dropout
--gelu_activation true                   # GELU instead of ReLU

## optimization
--batch_size 32                          # sequences per batch
--bptt 256                               # sequences length
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,weight_decay=0  # optimizer
--epoch_size 200000                      # number of sentences per epoch
--validation_metrics _valid_mlm_ppl      # validation metric (when to save the best model)
--stopping_criterion _valid_mlm_ppl,10   # end experiment if stopping criterion does not improve
```

### Train on XNLI from a pretrained model

You can now use the pretrained model for cross-lingual classification. To download a model trained with the command above on the MLM-TLM objective, run:

```
wget -c https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth
```

You can now fine-tune the pretrained model on XNLI, or on one of the English GLUE tasks:

```
python glue-xnli.py
--exp_name test_xnli_mlm_tlm             # experiment name
--dump_path ./dumped/                    # where to store the experiment
--model_path mlm_tlm_xnli15_1024.pth     # model location
--data_path ./data/processed/XLM15       # data location
--transfer_tasks XNLI,SST-2              # transfer tasks (XNLI or GLUE tasks)
--optimizer adam,lr=0.000005             # optimizer
--batch_size 8                           # batch size
--n_epochs 250                           # number of epochs
--epoch_size 20000                       # number of sentences per epoch
--max_len 256                            # max number of words in sentences
--max_vocab 95000                        # max number of words in vocab
```

## Frequently Asked Questions

### How can I run experiments on multiple GPUs?

XLM supports both multi-GPU and multi-node training, and was tested with up to 128 GPUs. To run an experiment with multiple GPUs on a single machine, simply replace `python train.py` in the commands above with:

```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
```

The multi-node is automatically handled by SLURM.

## References

Please cite [[1]](https://arxiv.org/abs/1901.07291) if you found the resources in this repository useful.

### Cross-lingual Language Model Pretraining

[1] G. Lample *, A. Conneau * [*Cross-lingual Language Model Pretraining*](https://arxiv.org/abs/1901.07291)

\* Equal contribution. Order has been determined with a coin flip.

```
@article{lample2019cross,
  title={Cross-lingual Language Model Pretraining},
  author={Lample, Guillaume and Conneau, Alexis},
  journal={arXiv preprint arXiv:1901.07291},
  year={2019}
}
```

### XNLI: Evaluating Cross-lingual Sentence Representations

[2] A. Conneau, G. Lample, R. Rinott, A. Williams, S. R. Bowman, H. Schwenk, V. Stoyanov [*XNLI: Evaluating Cross-lingual Sentence Representations*](https://arxiv.org/abs/1809.05053)

```
@inproceedings{conneau2018xnli,
  title={XNLI: Evaluating Cross-lingual Sentence Representations},
  author={Conneau, Alexis and Lample, Guillaume and Rinott, Ruty and Williams, Adina and Bowman, Samuel R and Schwenk, Holger and Stoyanov, Veselin},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2018}
}
```

### Phrase-Based \& Neural Unsupervised Machine Translation

[3] G. Lample, M. Ott, A. Conneau, L. Denoyer, MA. Ranzato [*Phrase-Based & Neural Unsupervised Machine Translation*](https://arxiv.org/abs/1804.07755)

```
@inproceedings{lample2018phrase,
  title={Phrase-Based \& Neural Unsupervised Machine Translation},
  author={Lample, Guillaume and Ott, Myle and Conneau, Alexis and Denoyer, Ludovic and Ranzato, Marc'Aurelio},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2018}
}
```

## License

See the [LICENSE](LICENSE) file for more details.
