# Tools

In `XLM/tools/`, you will need to install the following tools:

## Tokenizers

[Moses](https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer) tokenizer:
```
git clone https://github.com/moses-smt/mosesdecoder
```

Thai [PythaiNLP](https://github.com/PyThaiNLP/pythainlp) tokenizer:
```
pip install pythainlp
```

Japanese [KyTea](http://www.phontron.com/kytea) tokenizer:
```
wget http://www.phontron.com/kytea/download/kytea-0.4.7.tar.gz
tar -xzf kytea-0.4.7.tar.gz
cd kytea-0.4.7
./configure
make
make install
kytea --help
```

Chinese Stanford segmenter:
```
wget https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip
unzip stanford-segmenter-2018-10-16.zip
```

## fastBPE

```
git clone https://github.com/glample/fastBPE
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```
