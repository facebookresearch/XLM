# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch

from .pretrain import load_embeddings
from .transformer import DECODER_ONLY_PARAMS, TransformerModel, Embedding, Linear, N_MAX_POSITIONS  # , TRANSFORMER_LAYER_PARAMS
from .memory import HashingMemory

import IPython as ipy

logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.sample_alpha < 1
    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    # input sentence noise for DAE
    if len(params.ae_steps) == 0:
        assert params.word_shuffle == 0
        assert params.word_dropout == 0
        assert params.word_blank == 0
    else:
        assert params.word_shuffle == 0 or params.word_shuffle > 1
        assert 0 <= params.word_dropout < 1
        assert 0 <= params.word_blank < 1

    # model dimensions
    model_dim = params.model_dim if params.model_dim != -1 else params.emb_dim
    assert model_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # memory
    if params.use_memory:
        HashingMemory.check_params(params)
        s_enc = [x for x in params.mem_enc_positions.split(',') if x != '']
        s_dec = [x for x in params.mem_dec_positions.split(',') if x != '']
        assert len(s_enc) == len(set(s_enc))
        assert len(s_dec) == len(set(s_dec))
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_enc)
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_dec)
        params.mem_enc_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_enc]
        params.mem_dec_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_dec]
        assert len(params.mem_enc_positions) + len(params.mem_dec_positions) > 0
        assert len(params.mem_enc_positions) == 0 or 0 <= min([x[0] for x in params.mem_enc_positions]) <= max([x[0] for x in params.mem_enc_positions]) <= params.n_layers - 1
        assert len(params.mem_dec_positions) == 0 or 0 <= min([x[0] for x in params.mem_dec_positions]) <= max([x[0] for x in params.mem_dec_positions]) <= params.n_layers - 1

    # reload pretrained word embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    # reload a pretrained model
    if params.reload_model != '':
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            s = params.reload_model.split(',')
            assert len(s) == 2
            assert all([x == '' or os.path.isfile(x) for x in s])


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Loaded pretrained embs for %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


def build_model(params, dico):
    """
    Build model.
    """
    if params.encoder_only:
        # build
        model = TransformerModel(params, dico, is_encoder=True, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(model, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(params.local_rank))['model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            # # HACK to reload models with less layers
            # for i in range(12, 24):
            #     for k in TRANSFORMER_LAYER_PARAMS:
            #         k = k % i
            #         if k in model.state_dict() and k not in reloaded:
            #             logger.warning("Parameter %s not found. Ignoring ..." % k)
            #             reloaded[k] = model.state_dict()[k]

            # HERE: Re-initialize 'position_embeddings.weight', 'embeddings.weight'

            # Keep old parameters of embeddings for source language (English)
            if 'embeddings.weight' in reloaded:
                params.src_emb = reloaded['embeddings.weight'].clone().to(0)
            elif 'pred_layer.proj.head.weight' in reloaded:
                model.load_state_dict(reloaded)
                params.src_emb = model.pred_layer.embed_asm_input(inputs).clone().to(0)

            # "Reload" parameters of embeddings for target language (others) by creating new ones
            if params.reinit:
                reloaded['position_embeddings.weight'] = Embedding(N_MAX_POSITIONS, params.emb_dim).weight.data.half().cuda().to(0)
                if 'embeddings.weight' in reloaded:
                    reloaded['embeddings.weight'] = Embedding(params.n_words, params.emb_dim, padding_idx=params.pad_index).weight.data.half().cuda().to(0)
                elif params.asm_input:
                    reloaded['pred_layer.proj.head.weight'] = model.pred_layer.proj.head.weight.data.half().cuda().to(0)
                    reloaded['pred_layer.proj.head.bias'] = model.pred_layer.proj.head.bias.data.half().cuda().to(0)
                    for i in range(len(model.pred_layer.proj.cutoffs)-1):
                        reloaded[f'pred_layer.proj.tail.{i}.0.weight'] = model.pred_layer.proj.tail[i][0].weight.data.half().cuda().to(0)
                        reloaded[f'pred_layer.proj.tail.{i}.1.weight'] = model.pred_layer.proj.tail[i][1].weight.data.half().cuda().to(0)
                else:
                    raise ValueError("reloaded model does not have embeddings but params.asm_input is not set")
                if 'post_embed_proj.weight' in reloaded:
                    pep = Linear(params.emb_dim, params.model_dim)
                    reloaded['post_embed_proj.weight'] = pep.weight.data.half().cuda().to(0)
                    reloaded['post_embed_proj.bias'] = pep.bias.data.half().cuda().to(0)
                if 'post_embed_pos_proj.weight' in reloaded:
                    pepp = Linear(params.emb_dim, params.model_dim)
                    reloaded['post_embed_pos_proj.weight'] = pepp.weight.data.half().cuda().to(0)
                    reloaded['post_embed_pos_proj.bias'] = pepp.bias.data.half().cuda().to(0)
                if 'embeddings.weight' in reloaded and params.share_inout_emb:
                    # tie input/output embeddings
                    reloaded['pred_layer.proj.weight'] = reloaded['embeddings.weight'].clone().to(0)
                    reloaded['pred_layer.proj.bias'] = Linear(params.emb_dim, params.n_words, bias=True).bias.data.half().cuda().to(0)
            else:
                ipy.embed()
            model.load_state_dict(reloaded)

            if params.reload_emb != '':
                # load embeddings from file
                word2id, embeddings = load_embeddings(params.reload_emb, params)
                # rescale embeddings to match pretrained model. 
                # this is only approximate as some words don't have preloaded embs, but the initialized values should be close to the desired scale anyway
                loaded_mean = embeddings.abs().mean()
                pretrained_mean = params.src_emb.abs().mean()
                embeddings = embeddings/(loaded_mean/pretrained_mean)
                # update model
                set_pretrain_emb(model, dico, word2id, embeddings)

        logger.info("Model: {}".format(model))
        logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))

        return model.cuda()

    else:
        # build
        encoder = TransformerModel(params, dico, is_encoder=True, with_output=True)  # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
        decoder = TransformerModel(params, dico, is_encoder=False, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(encoder, dico, word2id, embeddings)
            set_pretrain_emb(decoder, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            enc_path, dec_path = params.reload_model.split(',')
            assert not (enc_path == '' and dec_path == '')

            # reload encoder
            if enc_path != '':
                logger.info("Reloading encoder from %s ..." % enc_path)
                enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
                if all([k.startswith('module.') for k in enc_reload.keys()]):
                    enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}
                encoder.load_state_dict(enc_reload)

            # reload decoder
            if dec_path != '':
                logger.info("Reloading decoder from %s ..." % dec_path)
                dec_reload = torch.load(dec_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                dec_reload = dec_reload['model' if 'model' in dec_reload else 'decoder']
                if all([k.startswith('module.') for k in dec_reload.keys()]):
                    dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}
                for i in range(params.n_layers):
                    for name in DECODER_ONLY_PARAMS:
                        if name % i not in dec_reload:
                            logger.warning("Parameter %s not found." % (name % i))
                            dec_reload[name % i] = decoder.state_dict()[name % i]
                decoder.load_state_dict(dec_reload)

        logger.debug("Encoder: {}".format(encoder))
        logger.debug("Decoder: {}".format(decoder))
        logger.info("Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        logger.info("Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))

        return encoder.cuda(), decoder.cuda()
