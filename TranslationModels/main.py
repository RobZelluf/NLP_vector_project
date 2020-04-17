import os
import numpy as np
import gensim
from gensim.test.utils import datapath as gensim_datapath
from gensim.models import KeyedVectors

import sys
sys.path.append("./../")

from TranslationModels.rnn_model import RNNModel
from TranslationModels.dataloader import tr_data_loader
from TranslationModels.const_vars import *
from TranslationModels.wvectors_tool import getVectorModel
from TranslationModels.transformer_model import TransformerModel


def extendPretrainedModel(model):
    length = model.vector_size
    try:
        model.get(SOS_token)
    except:
        model.add(SOS_token, np.random.normal(0, 0.01, length))
    try:
        model.get(EOS_token)
    except:
        model.add(EOS_token, np.random.normal(0, 0.01, length))
    try:
       model.get(UNK_token)
    except:
        model.add(UNK_token, np.random.normal(0, 0.01, length))
    return model

if __name__=="__main__":

    isTransformer = True
    train = False

    path_src_train_file = "./../data/train_data/OpenSubtitles.en-nl.en"
    path_tgt_train_file = "./../data/train_data/OpenSubtitles.en-nl.nl"

    path_nl_vw_model_bin = "./../data/vector_models/nl_d100.bin"
    path_en_vw_model_bin = "./../data/vector_models/en_d100.bin"

    print("+ preparing src vector model")
    vw_src_model = KeyedVectors.load_word2vec_format(path_en_vw_model_bin, binary=True)
    print("++ src vector model read")
    vw_src_model = extendPretrainedModel(vw_src_model)
    print("++ src vector model extended")

    print("+ preparing tgt vector model")
    vw_tgt_model = KeyedVectors.load_word2vec_format(path_nl_vw_model_bin, binary=True)
    print("++ tgt vector model read")
    vw_tgt_model = extendPretrainedModel(vw_tgt_model)
    print("++ tgt vector model extended")

    if not isTransformer:
        translation_model = RNNModel(
             src_vectorModel=vw_src_model,
             tgt_vectorModel=vw_tgt_model,
             encoder_save_path="./../data/translation_models/rnn_encoder_model.pth",
             decoder_save_path="./../data/translation_models/rnn_decoder_model.pth",
             hidden_size=1024)
    else:
        translation_model = TransformerModel(
             src_vectorModel=vw_src_model,
             tgt_vectorModel=vw_tgt_model,
             encoder_save_path="./../data/translation_models/tr_encoder_model.pth",
             decoder_save_path="./../data/translation_models/tr_decoder_model.pth",
             hidden_size=1024)

    if train:
        print("+ start TrNN training")
        translation_model.train(
             path_src_train_file,
             path_tgt_train_file,
             batch_size=4,
             iters=2,
             #device = "cuda:0")
             )

    print("+ Loading model")
    try:
        if not isTransformer:
            translation_model.load(
               encoder_path="./../data/translation_models/rnn_encoder_model.pth",
               decoder_path="./../data/translation_models/rnn_decoder_model.pth")
        else:
            translation_model.load(
               encoder_path="./../data/translation_models/tr_encoder_model.pth",
               decoder_path="./../data/translation_models/tr_decoder_model.pth")
    except Exception as e:
        print(e)
    else:
        print("++ loaded")

    print()
    tr_input = "I want a dog."
    tr_res = translation_model.translate(tr_input, True)
    print("+ Translation:")
    print("++ Input:", tr_input)
    print("++ Output:", tr_res)

    print()
    print("done!")
