import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence

from gensim.test.utils import datapath
from gensim.utils import chunkize_serial
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation, strip_tags, strip_multiple_whitespaces
from gensim.models.callbacks import CallbackAny2Vec

class tr_data_loader(object):
    def __init__(self, src_vectorModel, tgt_vectorModel,
                 filesrc, filetgt, batch_size,
                 sos_token, eos_token, unk_token, padding_value=None):
        self.filesrc = filesrc
        self.filetgt = filetgt
        self.batch_size = batch_size
        self.src_vm = src_vectorModel
        self.tgt_vm = tgt_vectorModel
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sos_token_index = self.src_vm.vocab.get(self.sos_token).index
        self.eos_token_index = self.src_vm.vocab.get(self.eos_token).index
        self.unk_token_index = self.src_vm.vocab.get(self.unk_token).index
        if padding_value is None:
            self.padding_value = self.eos_token_index
        else:
            self.padding_value = padding_value

    def collate(self, list_of_samples):
        """Merges a list of samples to form a mini-batch.

        Args:
          list_of_samples is a list of tuples (src_seq, tgt_seq):
              src_seq is of shape (src_seq_length)
              tgt_seq is of shape (tgt_seq_length)

        Returns:
          src_seqs of shape (max_src_seq_length, batch_size): Tensor of padded source sequences.
          src_mask of shape (max_src_seq_length, batch_size): Boolean tensor showing which elements of the
              src_seqs tensor should be ignored in computations (filled with PADDING_VALUE).
          tgt_seqs of shape (max_tgt_seq_length+1, batch_size): Tensor of padded target sequences.
        """
        src_samples, tgt_samples = list(zip(*list_of_samples))

        max_src_seq_length = max([s.size(0) for s in src_samples])
        src_out_dims = (max_src_seq_length, len(src_samples))
        src_seqs = src_samples[0].data.new(*src_out_dims).fill_(self.padding_value)
        src_mask = torch.ones(*src_out_dims, dtype=torch.bool)
        for i, src_tensor in enumerate(src_samples):
            length = src_tensor.size(0)
            src_seqs[:length, i] = src_tensor
            src_mask[:length, i] = False

        max_tgt_seq_length = max([s.size(0) for s in tgt_samples])
        tgt_out_dims = (1 + max_tgt_seq_length, len(tgt_samples))
        tgt_seqs = tgt_samples[0].data.new(*tgt_out_dims).fill_(self.padding_value)
        for i, tgt_tensor in enumerate(tgt_samples):
            length = tgt_tensor.size(0)
            tgt_seqs[0, i] = self.sos_token_index
            tgt_seqs[1:length + 1, i] = tgt_tensor

        return src_seqs, src_mask, tgt_seqs

    def __iter__(self):
        with open(self.filesrc) as file_src, open(self.filetgt) as file_tgt:
            lst = []
            i = 0
            for linesrc, linetgt in zip(file_src, file_tgt):
                linesrc = preprocess_string(linesrc, [strip_punctuation, strip_tags, strip_multiple_whitespaces])
                linetgt = preprocess_string(linetgt, [strip_punctuation, strip_tags, strip_multiple_whitespaces])

                linesrc = [*linesrc, self.eos_token]
                linetgt = [*linetgt, self.eos_token]

                linesrc_index = []
                for w in linesrc:
                    vw_index = self.src_vm.vocab.get(w)
                    if vw_index is None:
                        linesrc_index.append(self.unk_token_index)
                    else:
                        linesrc_index.append(vw_index.index)

                linetgt_index = []
                for w in linetgt:
                    vw_index = self.tgt_vm.vocab.get(w)
                    if vw_index is None:
                        linetgt_index.append(self.unk_token_index)
                    else:
                        linetgt_index.append(vw_index.index)

                linesrc_index = torch.tensor(linesrc_index).long()
                linetgt_index = torch.tensor(linetgt_index).long()

                lst.append((linesrc_index, linetgt_index))
                i += 1
                if i % self.batch_size == 0:
                    yield self.collate(lst)
                    lst = []
                    if i > 10:
                        break