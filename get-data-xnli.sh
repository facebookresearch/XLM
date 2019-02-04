# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

# data paths
MAIN_PATH=$PWD
OUTPATH=$PWD/data/xnli
XNLI_PATH=$PWD/data/xnli/XNLI-1.0
PROCESSED_PATH=$PWD/data/processed/XLM15
CODES_PATH=$MAIN_PATH/codes_xnli_15
VOCAB_PATH=$MAIN_PATH/vocab_xnli_15

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast

# install tools
./install-tools.sh

# create directories
mkdir -p $OUTPATH
mkdir -p $XNLI_PATH

# download data
if [ ! -d $OUTPATH/XNLI-MT-1.0 ]; then
  if [ ! -f $OUTPATH/XNLI-MT-1.0.zip ]; then
    wget -c https://www.nyu.edu/projects/bowman/xnli/XNLI-MT-1.0.zip -P $OUTPATH
  fi
  unzip $OUTPATH/XNLI-MT-1.0.zip -d $OUTPATH
fi
if [ ! -d $OUTPATH/XNLI-1.0 ]; then
  if [ ! -f $OUTPATH/XNLI-1.0.zip ]; then
    wget -c https://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip -P $OUTPATH
  fi
  unzip $OUTPATH/XNLI-1.0.zip -d $OUTPATH
fi

# English train set
echo "*** Preparing English train set ****"
cat $OUTPATH/XNLI-MT-1.0/multinli/multinli.train.en.tsv | sed 's/\tcontradictory/\tcontradiction/g' > $XNLI_PATH/en.train

# validation and test sets
for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do

  echo "*** Preparing $lg validation and test sets ***"
  echo -e "premise\thypo\tlabel" > $XNLI_PATH/$lg.valid
  echo -e "premise\thypo\tlabel" > $XNLI_PATH/$lg.test

  # label
  awk -v lg=$lg '$1==lg' $XNLI_PATH/xnli.dev.tsv  | cut -f2 > $XNLI_PATH/dev.f2
  awk -v lg=$lg '$1==lg' $XNLI_PATH/xnli.test.tsv | cut -f2 > $XNLI_PATH/test.f2

  # premise/hypothesis
  awk -v lg=$lg '$1==lg' $XNLI_PATH/xnli.dev.tsv  | cut -f7 | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $XNLI_PATH/dev.f7
  awk -v lg=$lg '$1==lg' $XNLI_PATH/xnli.dev.tsv  | cut -f8 | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $XNLI_PATH/dev.f8
  awk -v lg=$lg '$1==lg' $XNLI_PATH/xnli.test.tsv | cut -f7 | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $XNLI_PATH/test.f7
  awk -v lg=$lg '$1==lg' $XNLI_PATH/xnli.test.tsv | cut -f8 | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $XNLI_PATH/test.f8

  paste $XNLI_PATH/dev.f7  $XNLI_PATH/dev.f8  $XNLI_PATH/dev.f2  >> $XNLI_PATH/$lg.valid
  paste $XNLI_PATH/test.f7 $XNLI_PATH/test.f8 $XNLI_PATH/test.f2 >> $XNLI_PATH/$lg.test

  rm $XNLI_PATH/*.f2 $XNLI_PATH/*.f7 $XNLI_PATH/*.f8
done

mkdir -p $PROCESSED_PATH/eval/XNLI
rm $PROCESSED_PATH/eval/XNLI/*

# Get BPE codes and vocab
wget -c https://dl.fbaipublicfiles.com/XLM/codes_xnli_15 -P $MAIN_PATH
wget -c https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15 -P $MAIN_PATH

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
