# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import io
import numpy as np
import torch


logger = getLogger()


def load_fasttext_model(path):
    """
    Load a binarized fastText model.
    """
    try:
        import fastText
    except ImportError:
        raise Exception("Unable to import fastText. Please install fastText for Python: "
                        "https://github.com/facebookresearch/fastText")
    return fastText.load_model(path)


def read_txt_embeddings(path, params):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    _emb_dim_file = params.emb_dim
    with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
                continue
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            if word in word2id:
                logger.warning("Word \"%s\" found twice!" % word)
                continue
            if not vect.shape == (_emb_dim_file,):
                logger.warning("Invalid dimension (%i) for word \"%s\" in line %i."
                               % (vect.shape[0], word, i))
                continue
            assert vect.shape == (_emb_dim_file,)
            word2id[word] = len(word2id)
            vectors.append(vect[None])

    assert len(word2id) == len(vectors)
    logger.info("Loaded %i pretrained word embeddings from %s" % (len(vectors), path))

    # compute new vocabulary / embeddings
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()

    assert embeddings.size() == (len(word2id), params.emb_dim)
    return word2id, embeddings


def load_bin_embeddings(path, params):
    """
    Reload pretrained embeddings from a fastText binary file.
    """
    model = load_fasttext_model(path)
    assert model.get_dimension() == params.emb_dim
    words = model.get_labels()
    logger.info("Loaded binary model from %s" % path)

    # compute new vocabulary / embeddings
    embeddings = np.concatenate([model.get_word_vector(w)[None] for w in words], 0)
    embeddings = torch.from_numpy(embeddings).float()
    word2id = {w: i for i, w in enumerate(words)}
    logger.info("Generated embeddings for %i words." % len(words))

    assert embeddings.size() == (len(word2id), params.emb_dim)
    return word2id, embeddings


def load_embeddings(path, params):
    """
    Reload pretrained embeddings.
    """
    if path.endswith('.bin'):
        return load_bin_embeddings(path, params)
    else:
        return read_txt_embeddings(path, params)
