# XLM

PyTorch original implementation of [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291). Includes:
- [Monolingual language model pretraining (BERT)](#i-monolingual-language-model-pretraining-bert)
- [Cross-lingual language model pretraining (XLM)](#ii-cross-lingual-language-model-pretraining-xlm)
- [Applications: Supervised / Unsupervised MT (NMT / UNMT)](#iii-applications-supervised--unsupervised-mt)
- [Applications: Cross-lingual text classification (XNLI)](#iv-applications-cross-lingual-text-classification-xnli)
- [Product-Key Memory Layers (PKM)](#v-product-key-memory-layers-pkm)

**Update:** [**New models in 17 and 100 languages**](#pretrained-cross-lingual-language-models)

<br>
<br>

![Model](https://dl.fbaipublicfiles.com/XLM/xlm_figure.jpg)

<br>
<br>

XLM supports multi-GPU and multi-node training, and contains code for:
- **Language model pretraining**:
    - **Causal Language Model** (CLM)
    - **Masked Language Model** (MLM)
    - **Translation Language Model** (TLM)
- **GLUE** fine-tuning
- **XNLI** fine-tuning
- **Supervised / Unsupervised MT** training:
    - Denoising auto-encoder
    - Parallel data training
    - Online back-translation

## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (currently tested on version 0.4 and 1.0)
- [fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) (generate and apply BPE codes)
- [Moses](https://github.com/facebookresearch/XLM/tree/master/tools#tokenizers) (scripts to clean and tokenize text only - no installation required)
- [Apex](https://github.com/nvidia/apex#quick-start) (for fp16 training)


## I. Monolingual language model pretraining (BERT)
In what follows we explain how you can download and use our pretrained XLM (English-only) BERT model. Then we explain how you can train your own monolingual model, and how you can fine-tune it on the GLUE tasks.

### Pretrained English model
We provide our pretrained **XLM_en** English model, trained with the MLM objective.

| Languages        | Pretraining | Model                                                               | BPE codes                                                     | Vocabulary                                                     |
| ---------------- | ----------- |:-------------------------------------------------------------------:|:-------------------------------------------------------------:| --------------------------------------------------------------:|
| English          |     MLM     | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_en_2048.pth)         | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_en)      | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_en)    |

which obtains better performance than BERT (see the [GLUE benchmark](https://gluebenchmark.com/leaderboard)) while trained on the same data:

Model | Score | CoLA | SST2 | MRPC | STS-B | QQP | MNLI_m | MNLI_mm | QNLI | RTE | WNLI | AX
|:---: |:---: |:---: | :---: |:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
`BERT` | 80.5 | 60.5 | 94.9 | 89.3/85.4 | 87.6/86.5 | 72.1/89.3 | 86.7 | 85.9 | 92.7 | 70.1 | 65.1 | 39.6
`XLM_en` | **82.8** | **62.9** | **95.6** | **90.7/87.1** | **88.8/88.2** | **73.2/89.8** | **89.1** | **88.5** | **94.0** | **76.0** | **71.9** | **44.7**

If you want to **play around with the model and its representations**, just download the model and take a look at our **[ipython notebook](https://github.com/facebookresearch/XLM/blob/master/generate-embeddings.ipynb)** demo.

Our **XLM** PyTorch English model is trained on the same data than the pretrained **BERT** [TensorFlow](https://github.com/google-research/bert) model (Wikipedia + Toronto Book Corpus). Our implementation does not use the next-sentence prediction task and has only 12 layers but higher capacity (665M parameters). Overall, our model achieves a better performance than the original BERT on all GLUE tasks (cf. table above for comparison).

### Train your own monolingual BERT model
Now it what follows, we will explain how you can train a similar model on your own data.

### 1. Preparing the data
First, get the monolingual data (English Wikipedia, the [TBC corpus](https://yknzhu.wixsite.com/mbweb) is not hosted anymore).
```
# Download and tokenize Wikipedia data in 'data/wiki/en.{train,valid,test}'
# Note: the tokenization includes lower-casing and accent-removal
./get-data-wiki.sh en
```

[Install fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) and **learn BPE** vocabulary (with 30,000 codes here):
```
OUTPATH=data/processed/XLM_en/30k  # path where processed files will be stored
FASTBPE=tools/fastBPE/fast  # path to the fastBPE tool

# create output path
mkdir -p $OUTPATH

# learn bpe codes on the training set (or only use a subset of it)
$FASTBPE learnbpe 30000 data/wiki/txt/en.train > $OUTPATH/codes
```

Now **apply BPE** tokenization to train/valid/test files:
```
$FASTBPE applybpe $OUTPATH/train.en data/wiki/txt/en.train $OUTPATH/codes &
$FASTBPE applybpe $OUTPATH/valid.en data/wiki/txt/en.valid $OUTPATH/codes &
$FASTBPE applybpe $OUTPATH/test.en data/wiki/txt/en.test $OUTPATH/codes &
```

and get the post-BPE vocabulary:
```
cat $OUTPATH/train.en | $FASTBPE getvocab - > $OUTPATH/vocab &
```

**Binarize the data** to limit the size of the data we load in memory:
```
# This will create three files: $OUTPATH/{train,valid,test}.en.pth
# After that we're all set
python preprocess.py $OUTPATH/vocab $OUTPATH/train.en &
python preprocess.py $OUTPATH/vocab $OUTPATH/valid.en &
python preprocess.py $OUTPATH/vocab $OUTPATH/test.en &
```

### 2. Train the BERT model
Train your BERT model (without the next-sentence prediction task) on the preprocessed data:

```

python train.py

## main parameters
--exp_name xlm_en                          # experiment name
--dump_path ./dumped                       # where to store the experiment

## data location / training objective
--data_path $OUTPATH                       # data location
--lgs 'en'                                 # considered languages
--clm_steps ''                             # CLM objective (for training GPT-2 models)
--mlm_steps 'en'                           # MLM objective

## transformer parameters
--emb_dim 2048                             # embeddings / model dimension (2048 is big, reduce if only 16Gb of GPU memory)
--n_layers 12                              # number of layers
--n_heads 16                               # number of heads
--dropout 0.1                              # dropout
--attention_dropout 0.1                    # attention dropout
--gelu_activation true                     # GELU instead of ReLU

## optimization
--batch_size 32                            # sequences per batch
--bptt 256                                 # sequences length  (streams of 256 tokens)
--optimizer adam_inverse_sqrt,lr=0.00010,warmup_updates=30000,beta1=0.9,beta2=0.999,weight_decay=0.01,eps=0.000001  # optimizer (training is quite sensitive to this parameter)
--epoch_size 300000                        # number of sentences per epoch
--max_epoch 100000                         # max number of epochs (~infinite here)
--validation_metrics _valid_en_mlm_ppl     # validation metric (when to save the best model)
--stopping_criterion _valid_en_mlm_ppl,25  # stopping criterion (if criterion does not improve 25 times)
--fp16 true                                # use fp16 training

## bert parameters
--word_mask_keep_rand '0.8,0.1,0.1'        # bert masking probabilities
--word_pred '0.15'                         # predict 15 percent of the words

## There are other parameters that are not specified here (see train.py).
```

To [train with multiple GPUs](https://github.com/facebookresearch/XLM#how-can-i-run-experiments-on-multiple-gpus) use:
```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
```

**Tips**: Even when the validation perplexity plateaus, keep training your model. The larger the batch size the better (so using multiple GPUs will improve performance). Tuning the learning rate (e.g. [0.0001, 0.0002]) should help.

### 3. Fine-tune a pretrained model on GLUE tasks
Now that the model is pretrained, let's **finetune** it. First, download and preprocess the **GLUE tasks**:

```
# Download and tokenize GLUE tasks in 'data/glue/{MNLI,QNLI,SST-2,STS-B}'

./get-data-glue.sh

# Preprocessing should be the same than for training.
# If you removed lower-casing/accent-removal, it sould be reflected here as well.
```

and **prepare the GLUE data** using the codes and vocab:
```
# by default this script uses the BPE codes and vocab of pretrained XLM_en. Modify in script if needed.
./prepare-glue.sh
```

In addition to the **train.py** script, we provide a complementary script **glue-xnli.py** to fine-tune a model on either GLUE or XNLI.

You can now **fine-tune the pretrained model** on one of the English GLUE tasks using this config:

```
# Config used for fine-tuning our pretrained English BERT model (mlm_en_2048.pth)
python glue-xnli.py
--exp_name test_xlm_en_glue              # experiment name
--dump_path ./dumped                     # where to store the experiment
--model_path mlm_en_2048.pth             # model location
--data_path $OUTPATH                     # data location
--transfer_tasks MNLI-m,QNLI,SST-2       # transfer tasks (GLUE tasks)
--optimizer_e adam,lr=0.000025           # optimizer of projection (lr \in [0.000005, 0.000025, 0.000125])
--optimizer_p adam,lr=0.000025           # optimizer of projection (lr \in [0.000005, 0.000025, 0.000125])
--finetune_layers "0:_1"                 # fine-tune all layers
--batch_size 8                           # batch size (\in [4, 8])
--n_epochs 250                           # number of epochs
--epoch_size 20000                       # number of sentences per epoch (relatively small on purpose)
--max_len 256                            # max number of words in sentences
--max_vocab -1                           # max number of words in vocab
```
**Tips**: You should sweep over the batch size (4 and 8) and the learning rate (5e-6, 2.5e-5, 1.25e-4) parameters.

## II. Cross-lingual language model pretraining (XLM)

### Pretrained cross-lingual language models

We provide large pretrained models for the 15 languages of [XNLI](https://github.com/facebookresearch/XNLI), and two other models in [17 and 100 languages](#the-17-and-100-languages).

|Languages|Pretraining|Tokenization                          |  Model                                                              | BPE codes                                                            | Vocabulary                                                            |
|---------|-----------|--------------------------------------| ------------------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------- |
|15       |    MLM    |tokenize + lowercase + no accent + BPE| [Model](https://dl.fbaipublicfiles.com/XLM/mlm_xnli15_1024.pth)     | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_xnli_15) (80k)  | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15) (95k)  |
|15       | MLM + TLM |tokenize + lowercase + no accent + BPE| [Model](https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth) | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_xnli_15) (80k)  | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15) (95k)  |
|17       |    MLM    |tokenize + BPE                        | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_17_1280.pth)         | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_xnli_17) (175k) | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_17) (200k) |
|100      |    MLM    |tokenize + BPE                        | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_100_1280.pth)        | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_xnli_100) (175k)| [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_100) (200k)|

which obtains better performance than mBERT on the [XNLI cross-lingual classification task](https://arxiv.org/abs/1809.05053):

Model | lg | en | es | de | ar | zh | ur
|:---: |:---: |:---: | :---: |:---: | :---: | :---: | :---: |
`mBERT` | 102 | 81.4 | 74.3 | 70.5 | 62.1 | 63.8 | 58.3
`XLM (MLM)` | 15 | 83.2 | 76.3 | 74.2 | 68.5 | 71.9 | 63.4
`XLM (MLM+TLM)` | 15 | **85.0** | 78.9 | **77.8** | **73.1** | **76.5** | **67.3**
`XLM (MLM)` | 17 | 84.8 | **79.4** | 76.2 | 71.5 | 75 | - 
`XLM (MLM)` | 100 | 83.7 | 76.6 | 73.6 | 67.4 | 71.7 | 62.9

If you want to play around with the model and its representations, just download the model and take a look at our [ipython notebook](https://github.com/facebookresearch/XLM/blob/master/generate-embeddings.ipynb) demo.

#### The 17 and 100 Languages

The XLM-17 model includes these languages: en-fr-es-de-it-pt-nl-sv-pl-ru-ar-tr-zh-ja-ko-hi-vi

The XLM-100 model includes these languages: en-es-fr-de-zh-ru-pt-it-ar-ja-id-tr-nl-pl-simple-fa-vi-sv-ko-he-ro-no-hi-uk-cs-fi-hu-th-da-ca-el-bg-sr-ms-bn-hr-sl-zh_yue-az-sk-eo-ta-sh-lt-et-ml-la-bs-sq-arz-af-ka-mr-eu-tl-ang-gl-nn-ur-kk-be-hy-te-lv-mk-zh_classical-als-is-wuu-my-sco-mn-ceb-ast-cy-kn-br-an-gu-bar-uz-lb-ne-si-war-jv-ga-zh_min_nan-oc-ku-sw-nds-ckb-ia-yi-fy-scn-gan-tt-am

### Train your own XLM model with MLM or MLM+TLM
Now in what follows, we will explain how you can train an XLM model on your own data.

### 1. Preparing the data
**Monolingual data (MLM)**: Follow the same procedure as in [I.1](https://github.com/facebookresearch/XLM#1-preparing-the-data), and download multiple monolingual corpora, such as the Wikipedias.

Note that we provide a [tokenizer script](https://github.com/facebookresearch/XLM/blob/master/tools/tokenize.sh):

```
lg=en
cat my_file.$lg | ./tools/tokenize.sh $lg > my_tokenized_file.$lg &
```

**Parallel data (TLM)**: We provide download scripts for some language pairs in the *get-data-para.sh* script.
```
# Download and tokenize parallel data in 'data/wiki/para/en-zh.{en,zh}.{train,valid,test}'
./get-data-para.sh en-zh &
```

For other language pairs, look at the [OPUS collection](http://opus.nlpl.eu/), and modify the get-data-para.sh script [here)(https://github.com/facebookresearch/XLM/blob/master/get-data-para.sh#L179-L180) to add your own language pair.

Now create you training set for the BPE vocabulary, for instance by taking 100M sentences from each monolingua corpora.
```
# build the training set for BPE tokenization (50k codes)
OUTPATH=data/processed/XLM_en_zh/50k
mkdir -p $OUTPATH
shuf -r -n 10000000 data/wiki/train.en >> $OUTPATH/bpe.train
shuf -r -n 10000000 data/wiki/train.zh >> $OUTPATH/bpe.train
```
And learn the 50k BPE code as in the previous section on the bpe.train file. Apply BPE tokenization on the monolingual and parallel corpora, and binarize everything using *preprocess.py*:

```
pair=en-zh

for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    $FASTBPE applybpe $OUTPATH/$pair.$lg.$split data/wiki/para/$pair.$lg.$split $OUTPATH/codes
    python preprocess.py $OUTPATH/vocab $OUTPATH/$pair.$lg.$split
  done
done
```

### 2. Train the XLM model
Train your XLM (MLM only) on the preprocessed data:

```
python train.py

## main parameters
--exp_name xlm_en_zh                       # experiment name
--dump_path ./dumped                       # where to store the experiment

## data location / training objective
--data_path $OUTPATH                       # data location
--lgs 'en-zh'                              # considered languages
--clm_steps ''                             # CLM objective (for training GPT-2 models)
--mlm_steps 'en,zh'                        # MLM objective

## transformer parameters
--emb_dim 1024                             # embeddings / model dimension (2048 is big, reduce if only 16Gb of GPU memory)
--n_layers 12                              # number of layers
--n_heads 16                               # number of heads
--dropout 0.1                              # dropout
--attention_dropout 0.1                    # attention dropout
--gelu_activation true                     # GELU instead of ReLU

## optimization
--batch_size 32                            # sequences per batch
--bptt 256                                 # sequences length  (streams of 256 tokens)
--optimizer adam,lr=0.0001                 # optimizer (training is quite sensitive to this parameter)
--epoch_size 300000                        # number of sentences per epoch
--max_epoch 100000                         # max number of epochs (~infinite here)
--validation_metrics _valid_mlm_ppl        # validation metric (when to save the best model)
--stopping_criterion _valid_mlm_ppl,25     # stopping criterion (if criterion does not improve 25 times)
--fp16 true                                # use fp16 training

## There are other parameters that are not specified here (see [here](https://github.com/facebookresearch/XLM/blob/master/train.py#L24-L198)).
```

Here the validation metrics *_valid_mlm_ppl* is the average of MLM perplexities.

**MLM+TLM model**: If you want to **add TLM on top of MLM**, just add "en-zh" language pair in mlm_steps:
```
--mlm_steps 'en,zh,en-zh'                  # MLM objective
```

**Tips**: You can also pretrain your model with MLM-only, and then continue training with MLM+TLM with the *--reload_model* parameter.


### 3. Fine-tune XLM models (Applications, see below)

Cross-lingual language model (XLM) provides a strong pretraining method for cross-lingual understanding (XLU) tasks. In what follows, we present applications to machine translation (unsupervised and supervised) and cross-lingual classification (XNLI).


## III. Applications: Supervised / Unsupervised MT

XLMs can be used as a pretraining method for unsupervised or supervised neural machine translation.

### Pretrained XLM(MLM) models
The English-French, English-German and English-Romanian models are the ones we used in the paper for MT pretraining. They are trained with monolingual data only, with the MLM objective. If you use these models, you should use the same data preprocessing / BPE codes to preprocess your data. See the preprocessing commands in [get-data-nmt.sh](https://github.com/facebookresearch/XLM/blob/master/get-data-nmt.sh).

| Languages        | Pretraining | Model                                                               | BPE codes                                                     | Vocabulary                                                     |
| ---------------- | ----------- |:-------------------------------------------------------------------:|:-------------------------------------------------------------:| --------------------------------------------------------------:|
| English-French   |     MLM     | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_enfr_1024.pth)       | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_enfr)    | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_enfr)    |
| English-German   |     MLM     | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_ende_1024.pth)       | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_ende)    | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_ende)    |
| English-Romanian |     MLM     | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_enro_1024.pth)       | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_enro)    | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_enro)    |


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

## IV. Applications: Cross-lingual text classification (XNLI)
XLMs can be used to build cross-lingual classifiers. After fine-tuning an XLM model on an English training corpus for instance (e.g. of sentiment analysis, natural language inference), the model is still able to make accurate predictions at test time in other languages, for which there is very little or no training data. This approach is usually referred to as "zero-shot cross-lingual classification".

### Get the right tokenizers

Before running the scripts below, make sure you download the tokenizers from the [tools/](https://github.com/facebookresearch/XLM/tree/master/tools) directory.

### Download / preprocess monolingual data

Follow a similar approach than in section 1 for the 15 languages:
```
for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do
  ./get-data-wiki.sh $lg
done
```

Downloading the Wikipedia dumps make take several hours. The *get-data-wiki.sh* script will automatically download Wikipedia dumps, extract raw sentences, clean and tokenize them. Note that in our experiments we also concatenated the [Toronto Book Corpus](http://yknzhu.wixsite.com/mbweb) to the English Wikipedia, but this dataset is no longer hosted.

For Chinese and Thai you will need a special tokenizer that you can install using the commands below. For all other languages, the data will be tokenized with Moses scripts.

```
# Thai - https://github.com/PyThaiNLP/pythainlp
pip install pythainlp

# Chinese
cd tools/
wget https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip
unzip stanford-segmenter-2018-10-16.zip
```

### Download parallel data

This script will download and tokenize the parallel data used for the TLM objective:

```
lg_pairs="ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh"
for lg_pair in $lg_pairs; do
  ./get-data-para.sh $lg_pair
done
```

### Apply BPE and binarize
Apply BPE and binarize data similar to section 2.

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

### Download XNLI data

This script will download and tokenize the XNLI corpus:
```
./get-data-xnli.sh
```

### Preprocess data
This script will apply BPE using the XNLI15 bpe codes, and binarize data.
```
./prepare-xnli.sh
```

### Fine-tune your XLM model on cross-lingual classification (XNLI)

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
--optimizer_e adam,lr=0.000025           # optimizer of projection (lr \in [0.000005, 0.000025, 0.000125])
--optimizer_p adam,lr=0.000025           # optimizer of projection (lr \in [0.000005, 0.000025, 0.000125])
--finetune_layers "0:_1"                 # fine-tune all layers
--batch_size 8                           # batch size (\in [4, 8])
--n_epochs 250                           # number of epochs
--epoch_size 20000                       # number of sentences per epoch
--max_len 256                            # max number of words in sentences
--max_vocab 95000                        # max number of words in vocab
```

## V. Product-Key Memory Layers (PKM)

XLM also implements the Product-Key Memory layer (PKM) described in [[4]](https://arxiv.org/abs/1907.05242). To add a memory in (for instance) the layers 4 and 7 of an encoder, you can simply provide `--use_memory true --mem_enc_positions 4,7` as argument of `train.py` (and similarly for `--mem_dec_positions` and the decoder). All memory layer parameters can be found [here](https://github.com/facebookresearch/XLM/blob/master/src/model/memory/memory.py#L225).
A minimalist and simple implementation of the PKM layer, that uses the same configuration as in the paper, can be found in this **[ipython notebook](https://github.com/facebookresearch/XLM/blob/master/PKM-layer.ipynb)**.


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

### Large Memory Layers with Product Keys

[4] G. Lample, A. Sablayrolles, MA. Ranzato, L. Denoyer, H. Jégou [*Large Memory Layers with Product Keys*](https://arxiv.org/abs/1907.05242)

```
@article{lample2019large,
  title={Large Memory Layers with Product Keys},
  author={Lample, Guillaume and Sablayrolles, Alexandre and Ranzato, Marc'Aurelio and Denoyer, Ludovic and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:1907.05242},
  year={2019}
}
```

## License

See the [LICENSE](LICENSE) file for more details.
