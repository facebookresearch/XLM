# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# This script is meant to prepare data to reproduce XNLI experiments
# Just modify the "code" and "vocab" path for your own model
#

set -e

pair=$1  # input language pair

# data paths
MAIN_PATH=$PWD
PARA_PATH=$PWD/data/para
TOOLS_PATH=$PWD/tools
WIKI_PATH=$PWD/data/wiki
XNLI_PATH=$PWD/data/xnli/XNLI-1.0
PROCESSED_PATH=$PWD/data/processed/XLM15
CODES_PATH=$MAIN_PATH/codes_xnli_15
VOCAB_PATH=$MAIN_PATH/vocab_xnli_15
FASTBPE=$TOOLS_PATH/fastBPE/fast


# Get BPE codes and vocab
wget -c https://dl.fbaipublicfiles.com/XLM/codes_xnli_15 -P $MAIN_PATH
wget -c https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15 -P $MAIN_PATH


## Prepare monolingual data
# apply BPE codes and binarize the monolingual corpora
for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do
    for split in train valid test; do
    $FASTBPE applybpe $PROCESSED_PATH/$lg.$split $WIKI_PATH/txt/$lg.$split $CODES_PATH
    python preprocess.py $VOCAB_PATH $PROCESSED_PATH/$lg.$split
    done
done

## Prepare parallel data
# apply BPE codes and binarize the parallel corpora
for pair in ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh; do
    for lg in $(echo $pair | sed -e 's/\-/ /g'); do
        for split in train valid test; do
            $FASTBPE applybpe $PROCESSED_PATH/$pair.$lg.$split $PARA_PATH/$pair.$lg.$split $CODES_PATH
            python preprocess.py $VOCAB_PATH $PROCESSED_PATH/$pair.$lg.$split
        done
    done
done

## Prepare XNLI data
rm -rf $PROCESSED_PATH/eval/XNLI
mkdir -p $PROCESSED_PATH/eval/XNLI
# apply BPE codes and binarize the XNLI corpora
for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do
  for splt in train valid test; do
    if [ "$splt" = "train" ] && [ "$lg" != "en" ]; then
      continue
    fi
    sed '1d' $XNLI_PATH/${lg}.${splt} | cut -f1 > $PROCESSED_PATH/eval/XNLI/f1.tok
    sed '1d' $XNLI_PATH/${lg}.${splt} | cut -f2 > $PROCESSED_PATH/eval/XNLI/f2.tok
    sed '1d' $XNLI_PATH/${lg}.${splt} | cut -f3 > $PROCESSED_PATH/eval/XNLI/${splt}.label.${lg}

    $FASTBPE applybpe $PROCESSED_PATH/eval/XNLI/${splt}.s1.${lg} $PROCESSED_PATH/eval/XNLI/f1.tok ${CODES_PATH}
    $FASTBPE applybpe $PROCESSED_PATH/eval/XNLI/${splt}.s2.${lg} $PROCESSED_PATH/eval/XNLI/f2.tok ${CODES_PATH}

    python preprocess.py ${VOCAB_PATH} $PROCESSED_PATH/eval/XNLI/${splt}.s1.${lg}
    python preprocess.py ${VOCAB_PATH} $PROCESSED_PATH/eval/XNLI/${splt}.s2.${lg}

    rm $PROCESSED_PATH/eval/XNLI/*.tok*
  done
done