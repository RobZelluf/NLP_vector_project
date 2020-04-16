import os
import numpy as np
import gensim
from gensim.test.utils import datapath as gensim_datapath
from gensim.models import KeyedVectors

from TransformerModel.transformer_dataloader import tr_data_loader
from TransformerModel.const_vars import *
from TransformerModel.wvectors_tool import getVectorModel
from TransformerModel.transformer_model import TransformerModel

def extendPretrainedModel(model):
    length = EMBEDDING_VECTOR_SIZE
    model.add(SOS_token, np.random.normal(0, 0.01, length))
    model.add(EOS_token, np.random.normal(0, 0.01, length))
    model.add(UNK_token, np.random.normal(0, 0.01, length))
    return model

if __name__=="__main__":

    path_src_train_file = "./../tmp/en-nl.txt/OpenSubtitles.en-nl.en"
    path_tgt_train_file = "./../tmp/en-nl.txt/OpenSubtitles.en-nl.nl"

    path_nl_vw_model_bin = "./../tmp/vector_models/39/model.bin"
    path_en_vw_model_bin = "./../tmp/vector_models/40/model.bin"

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

    # vw_src_model = getVectorModel(True, kind='test')
    # print("src vw")
    # vw_tgt_model = getVectorModel(True, kind='test')
    # print("tgt vw")

    translation_model = TransformerModel(
        src_vectorModel=vw_src_model,
        tgt_vectorModel=vw_tgt_model,
        encoder_save_path="tr_encoder_model.pth",
        decoder_save_path="tr_decoder_model.pth",
        hidden_size=1024)

    # print("+ start TrNN training")
    # translation_model.train(
    #     path_src_train_file,
    #     path_tgt_train_file,
    #     batch_size=4,
    #     iters=2)

    print("+ Loading model")
    translation_model.load(
        encoder_path="tr_encoder_model.pth",
        decoder_path="tr_decoder_model.pth")
    print("++ loaded")

    print()
    tr_input = "I want a dog."
    tr_res = translation_model.translate(tr_input)
    print("+ Translation:")
    print("++ Input:", tr_input)
    print("++ Output:", tr_res)

    print()
    print("done!")
