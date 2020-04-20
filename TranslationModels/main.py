import os
import os.path
import numpy as np
import gensim
from gensim.test.utils import datapath as gensim_datapath
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors

import sys
sys.path.append(os.path.abspath(os.path.normpath(os.path.join(__file__, "./../../"))))
import argparse

from TranslationModels.const_vars import *
from TranslationModels.rnn_model import RNNModel
from TranslationModels.dataloader import tr_data_loader
from TranslationModels.transformer_model import TransformerModel


def extendPretrainedModel(model):
    length = model.vector_size
    try:
        model.get_vector(SOS_token)
    except:
        model.add(SOS_token, np.random.normal(0, 0.01, length))
    try:
        model.get_vector(EOS_token)
    except:
        model.add(EOS_token, np.random.normal(0, 0.01, length))
    try:
       model.get_vector(UNK_token)
    except:
        model.add(UNK_token, np.random.normal(0, 0.01, length))
    return model

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', action = 'store_true', help='Should training be performed.')
    parser.add_argument('--type', type = str, choices = ['rnn', 'tr'], help='Type of translation model.', default = 'tr')

    #parser.add_argument('--extend', '-e', action = 'store_true', help='Should the word vector space be extended with SOS, EOS and UNK tokens.')
    parser.add_argument('--src', choices = ['en', 'nl', 'ru'], help='Source language for translation.', default = 'en')
    parser.add_argument('--tgt', choices = ['en', 'nl', 'ru'], help='Target language for translation.', default = 'nl')

    parser.add_argument('--source_vm', type = str, help='Word vectors for the source language, filename in the data/vector_models folder.', required = True)
    parser.add_argument('--target_vm', type = str, help='Paired corpus in the target language, filename in the data/vector_models folder.', required = True)


    parser.add_argument('--hidden_size', type = int, help='', default = 256)
    parser.add_argument('--max_batches', '-m', type = int, help='Maximum number of batches.', default = None)
    parser.add_argument('--batch_size', '-b', type = int, help='Batch size.', default = 4)
    parser.add_argument('--iters', '-i', type = int, help='Number of iterations.', default = 30)
    parser.add_argument('--gpu', '-g', action = 'store_true', help='Should training be done on GPU.')
    parser.add_argument('--unfiltered', '-u', action = 'store_const', help='Use unfiltered data.', const = '', default = '_filtered')

    parser.add_argument('--target', type = str, help='Sentence to translate.', default = 'I want a dog')

    args = parser.parse_args()

    if args.src == args.tgt:
        print('Source and target language identical!')
        sys.exit()


    path_src_train_file = './../data/train_data/' + min(args.src, args.tgt) + '_' + max(args.src, args.tgt) + args.unfiltered + '/' + args.src + '_train.txt' 
    path_tgt_train_file = './../data/train_data/' + min(args.src, args.tgt) + '_' + max(args.src, args.tgt) + args.unfiltered + '/' + args.tgt + '_train.txt' 

    path_src_vw_model_bin = './../data/vector_models/' + args.source_vm + '.bin'
    path_tgt_vw_model_bin = './../data/vector_models/' + args.target_vm + '.bin'


    if not all([os.path.isfile(fname) for fname in [path_src_train_file, path_tgt_train_file, path_src_vw_model_bin, path_tgt_vw_model_bin]]):
        print('Some of the files given do not exist, perhaps check defaults!')
        sys.exit()

    print('+ preparing src vector model')
    if "ft" in args.source_vm:
        vw_src_model = FastTextKeyedVectors.load(path_src_vw_model_bin)
    else:
        vw_src_model = KeyedVectors.load_word2vec_format(path_src_vw_model_bin, binary=True)
    print('++ src vector model read')
    vw_src_model = extendPretrainedModel(vw_src_model)
    print('++ src vector model extended')

    print('+ preparing tgt vector model')
    if "ft" in args.target_vm:
        vw_tgt_model = FastTextKeyedVectors.load(path_tgt_vw_model_bin)
    else:
        vw_tgt_model = KeyedVectors.load_word2vec_format(path_tgt_vw_model_bin, binary=True)
    print('++ tgt vector model read')
    vw_tgt_model = extendPretrainedModel(vw_tgt_model)
    print('++ tgt vector model extended')

    translation_models_path = './../data/translation_models/'
    if not os.path.exists(translation_models_path):
        os.makedirs(translation_models_path)

    enc_path = translation_models_path + args.type + '_encoder_' + args.src + '_' + args.tgt + '_VM_' + args.source_vm + '_VM_' + args.target_vm + '.pth'
    dec_path = translation_models_path + args.type + '_decoder_' + args.src + '_' + args.tgt + '_VM_' + args.source_vm + '_VM_' + args.target_vm + '.pth'

    if args.type == 'rnn':
        translation_model = RNNModel(
             src_vectorModel=vw_src_model,
             tgt_vectorModel=vw_tgt_model,
             encoder_save_path=enc_path,
             decoder_save_path=dec_path,
             hidden_size=args.hidden_size)
    else:
        translation_model = TransformerModel(
             src_vectorModel=vw_src_model,
             tgt_vectorModel=vw_tgt_model,
             encoder_save_path=enc_path,
             decoder_save_path=dec_path,
             hidden_size=args.hidden_size)

    if args.train:
        print('+ start TrNN training')
        translation_model.train(
             path_src_train_file,
             path_tgt_train_file,
             batch_size=args.batch_size,
             iters=args.iters,
             max_batches = args.max_batches,
             device = 'cuda:0' if args.gpu else 'cpu')
             

    '''print('+ Loading model')
                try:
                    if args.type == 'rnn':
                        translation_model.load(
                           encoder_path=enc_path,
                           decoder_path=dec_path)
                    else:
                        translation_model.load(
                           encoder_path=enc_path,
                           decoder_path=dec_path)
                except Exception as e:
                    print(e)
                else:
                    print('++ loaded')'''

    print()
    tr_input = args.target # 'I want a dog.'
    tr_res = translation_model.translate(tr_input, True)
    print('+ Translation:')
    print('++ Input:', tr_input)
    print('++ Output:', tr_res)

    print()
    print('done!')
