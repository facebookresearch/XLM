# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

# data paths
MAIN_PATH=$PWD
OUTPATH=$PWD/data/glue
URLPATH=https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2F

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py


# install tools
./install-tools.sh

# create directories
# rm -r $OUTPATH
mkdir -p $OUTPATH


# SST-2
if [ ! -d $OUTPATH/SST-2 ]; then
    if [ ! -f $OUTPATH/SST-2zip ]; then
        wget -c "${URLPATH}SST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8" -P $OUTPATH
    fi
    unzip $OUTPATH/*SST-2* -d $OUTPATH
    for split in train dev
    do
      sed '1d' $OUTPATH/SST-2/${split}.tsv | cut -f1 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR > $OUTPATH/SST-2/${split}.x
      sed '1d' $OUTPATH/SST-2/${split}.tsv | cut -f2 > $OUTPATH/SST-2/${split}.y
      paste $OUTPATH/SST-2/${split}.x $OUTPATH/SST-2/${split}.y > $OUTPATH/SST-2/${split}.xlm.tsv
      rm $OUTPATH/SST-2/${split}.x $OUTPATH/SST-2/${split}.y
    done
    sed '1d' $OUTPATH/SST-2/test.tsv | cut -f2 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR > $OUTPATH/SST-2/test.xlm.tsv
    rm $OUTPATH/*SST-2.zip* 

fi

# SST-B
if [ ! -d $OUTPATH/STS-B ]; then
    if [ ! -f $OUTPATH/STS-B.zip ]; then
        wget -c "${URLPATH}STS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5" -P $OUTPATH
    fi
    unzip $OUTPATH/*STS-B* -d $OUTPATH
    for split in train dev test
    do
      sed '1d' $OUTPATH/STS-B/${split}.tsv | cut -f8 | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $OUTPATH/STS-B/${split}.x1
      sed '1d' $OUTPATH/STS-B/${split}.tsv | cut -f9 | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $OUTPATH/STS-B/${split}.x2
      if [ "$split" != "test" ]; then
        sed '1d' $OUTPATH/STS-B/${split}.tsv | cut -f10 > $OUTPATH/STS-B/${split}.y
        paste $OUTPATH/STS-B/${split}.x1 $OUTPATH/STS-B/${split}.x2 $OUTPATH/STS-B/${split}.y > $OUTPATH/STS-B/${split}.xlm.tsv
        rm $OUTPATH/STS-B/${split}.x1 $OUTPATH/STS-B/${split}.x2 $OUTPATH/STS-B/${split}.y
      else
        paste $OUTPATH/STS-B/${split}.x1 $OUTPATH/STS-B/${split}.x2 > $OUTPATH/STS-B/${split}.xlm.tsv
        rm $OUTPATH/STS-B/${split}.x1 $OUTPATH/STS-B/${split}.x2
      fi
    done
    rm $OUTPATH/*STS-B.zip* 

fi

# MNLI
if [ ! -d $OUTPATH/MNLI ]; then
    if [ ! -f $OUTPATH/MNLI.zip ]; then
        wget -c "${URLPATH}MNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce" -P $OUTPATH
    fi
    unzip $OUTPATH/*MNLI* -d $OUTPATH
    mv $OUTPATH/MNLI/dev_matched.tsv $OUTPATH/MNLI/dev.tsv
    mv $OUTPATH/MNLI/test_matched.tsv $OUTPATH/MNLI/test.tsv
    for split in train dev test
    do
      sed '1d' $OUTPATH/MNLI/${split}.tsv | cut -f9 | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $OUTPATH/MNLI/${split}.x1
      sed '1d' $OUTPATH/MNLI/${split}.tsv | cut -f10 | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $OUTPATH/MNLI/${split}.x2
      sed '1d' $OUTPATH/MNLI/${split}.tsv | cut -f12 > $OUTPATH/MNLI/${split}.y
      paste $OUTPATH/MNLI/${split}.x1 $OUTPATH/MNLI/${split}.x2 $OUTPATH/MNLI/${split}.y > $OUTPATH/MNLI/${split}.xlm.tsv
      rm $OUTPATH/MNLI/${split}.x1 $OUTPATH/MNLI/${split}.x2 $OUTPATH/MNLI/${split}.y
    done
    rm $OUTPATH/*MNLI.zip*
    mv $OUTPATH/MNLI $OUTPATH/MNLI-m

fi

# QNLI
if [ ! -d $OUTPATH/QNLI ]; then
    if [ ! -f $OUTPATH/QNLIv2.zip ]; then
        wget -c "${URLPATH}QNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601" -P $OUTPATH
    fi
    unzip $OUTPATH/*QNLIv2* -d $OUTPATH
    for split in train dev test
    do
      sed '1d' $OUTPATH/QNLI/${split}.tsv | cut -f2 | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $OUTPATH/QNLI/${split}.x1
      sed '1d' $OUTPATH/QNLI/${split}.tsv | cut -f3 | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $OUTPATH/QNLI/${split}.x2
      if [ "$split" != "test" ]; then
        sed '1d' $OUTPATH/QNLI/${split}.tsv | cut -f4 > $OUTPATH/QNLI/${split}.y
        paste $OUTPATH/QNLI/${split}.x1 $OUTPATH/QNLI/${split}.x2 $OUTPATH/QNLI/${split}.y > $OUTPATH/QNLI/${split}.xlm.tsv
        rm $OUTPATH/QNLI/${split}.x1 $OUTPATH/QNLI/${split}.x2 $OUTPATH/QNLI/${split}.y
      else
        paste $OUTPATH/QNLI/${split}.x1 $OUTPATH/QNLI/${split}.x2 > $OUTPATH/QNLI/${split}.xlm.tsv
        rm $OUTPATH/QNLI/${split}.x1 $OUTPATH/QNLI/${split}.x2
      fi
    done
    rm $OUTPATH/*QNLIv2.zip* 

fi

# QQP
if [ ! -d $OUTPATH/QQP ]; then
    if [ ! -f $OUTPATH/QQP.zip ]; then
        wget -c "${URLPATH}QQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5" -P $OUTPATH
    fi
    unzip $OUTPATH/*QQP* -d $OUTPATH
    for split in train dev test
    do
      if [ "$split" != "test" ]; then
        sed '1d' $OUTPATH/QQP/${split}.tsv | cut -f4 | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $OUTPATH/QQP/${split}.x1
        sed '1d' $OUTPATH/QQP/${split}.tsv | cut -f5 | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $OUTPATH/QQP/${split}.x2
        sed '1d' $OUTPATH/QQP/${split}.tsv | cut -f6 > $OUTPATH/QQP/${split}.y
        paste $OUTPATH/QQP/${split}.x1 $OUTPATH/QQP/${split}.x2 $OUTPATH/QQP/${split}.y > $OUTPATH/QQP/${split}.xlm.tsv
        rm $OUTPATH/QQP/${split}.x1 $OUTPATH/QQP/${split}.x2 $OUTPATH/QQP/${split}.y
      else
        sed '1d' $OUTPATH/QQP/${split}.tsv | cut -f2 | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $OUTPATH/QQP/${split}.x1
        sed '1d' $OUTPATH/QQP/${split}.tsv | cut -f3 | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $OUTPATH/QQP/${split}.x2
        paste $OUTPATH/QQP/${split}.x1 $OUTPATH/QQP/${split}.x2 > $OUTPATH/QQP/${split}.xlm.tsv
        rm $OUTPATH/QQP/${split}.x1 $OUTPATH/QQP/${split}.x2
      fi
    done
    rm $OUTPATH/*QQP.zip* 

fi

