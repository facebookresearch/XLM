# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Usage: ./get-data-xnli.sh
#

set -e

# data paths
MAIN_PATH=$PWD
OUTPATH=$PWD/data/xnli
XNLI_PATH=$PWD/data/xnli/XNLI-1.0

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py

# install tools
./install-tools.sh

# create directories
mkdir -p $OUTPATH

# download data
if [ ! -d $OUTPATH/XNLI-MT-1.0 ]; then
  if [ ! -f $OUTPATH/XNLI-MT-1.0.zip ]; then
    wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip -P $OUTPATH
  fi
  unzip $OUTPATH/XNLI-MT-1.0.zip -d $OUTPATH
fi
if [ ! -d $OUTPATH/XNLI-1.0 ]; then
  if [ ! -f $OUTPATH/XNLI-1.0.zip ]; then
    wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip -P $OUTPATH
  fi
  unzip $OUTPATH/XNLI-1.0.zip -d $OUTPATH
fi

# English train set
echo "*** Preparing English train set ****"
echo -e "premise\thypo\tlabel" > $XNLI_PATH/en.train
sed '1d'  $OUTPATH/XNLI-MT-1.0/multinli/multinli.train.en.tsv | cut -f1 | python $LOWER_REMOVE_ACCENT > $XNLI_PATH/train.f1
sed '1d'  $OUTPATH/XNLI-MT-1.0/multinli/multinli.train.en.tsv | cut -f2 | python $LOWER_REMOVE_ACCENT > $XNLI_PATH/train.f2
sed '1d'  $OUTPATH/XNLI-MT-1.0/multinli/multinli.train.en.tsv | cut -f3 | sed 's/contradictory/contradiction/g' > $XNLI_PATH/train.f3
paste $XNLI_PATH/train.f1 $XNLI_PATH/train.f2 $XNLI_PATH/train.f3 >> $XNLI_PATH/en.train

rm $XNLI_PATH/train.f1 $XNLI_PATH/train.f2 $XNLI_PATH/train.f3


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
