import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation, strip_tags, strip_multiple_whitespaces

import TransformerModel.tr_model_utils as tr
from TransformerModel.transformer_dataloader import tr_data_loader
from TransformerModel.const_vars import *


def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class EncoderBlock(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          n_features: Number of input and output features.
          n_heads: Number of attention heads in the Multi-Head Attention.
          n_hidden: Number of hidden units in the Feedforward (MLP) block.
          dropout: Dropout rate after the first layer of the MLP and the two skip connections.
        """
        super(EncoderBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(n_features, n_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(n_features)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(n_features)
        self.dropout_2 = nn.Dropout(dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features),
        )

    def forward(self, x, mask):
        """
        Args:
          x of shape (max_seq_length, batch_size, n_features): Input sequences.
          mask of shape (batch_size, max_seq_length): Boolean tensor indicating which elements of the input
              sequences should be ignored.
        Returns:
          z of shape (max_seq_length, batch_size, n_features): Encoded input sequence.
        Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
        """
        attn_output, attn_output_weights = self.multihead_attn(x, x, x, key_padding_mask=mask)
        x = self.layer_norm1(x + self.dropout_1(attn_output))
        pwff = self.position_wise_feed_forward(x)
        z = self.layer_norm2(x + self.dropout_2(pwff))
        return z


class Encoder(nn.Module):
    def __init__(self, embedding_vectors, n_blocks, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          embedding_vectors: Vector matrix from gensim pretrained vectors, accessing by model.vectors.
          n_blocks: Number of EncoderBlock blocks.
          n_heads: Number of attention heads inside the EncoderBlock.
          n_hidden: Number of hidden units in the Feedforward block of EncoderBlock.
          dropout: Dropout level used in EncoderBlock.

          # src_vocab_size: Number of words in the source vocabulary.
          # n_features: Number of features to be used for word embedding and further in all layers of the encoder.
        """
        super(Encoder, self).__init__()
        # self.embedding = nn.Embedding(src_vocab_size, n_features)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_vectors))
        # src_vocab_size = self.embedding.num_embeddings
        n_features = self.embedding.embedding_dim
        self.n_blocks = n_blocks

        self.pos_encoding = tr.PositionalEncoding(n_features, dropout=dropout, max_len=MAX_LENGTH)
        encoder_blocks_list = []
        for i in range(self.n_blocks):
            encoder_blocks_list.append(EncoderBlock(n_features=n_features,
                                                    n_heads=n_heads,
                                                    n_hidden=n_hidden,
                                                    dropout=dropout))
        self.encoder_blocks = nn.ModuleList(encoder_blocks_list)

    def forward(self, x, mask):
        """
        Args:
          x of shape (max_seq_length, batch_size): LongTensor with the input sequences.
          mask of shape (batch_size, max_seq_length): BoolTensor indicating which elements should be ignored.
        Returns:
          z of shape (max_seq_length, batch_size, n_features): Encoded input sequence.
        Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
        """
        z = self.pos_encoding(self.embedding(x))
        for i in range(self.n_blocks):
            z = self.encoder_blocks[i](z, mask=mask)
        return z


class DecoderBlock(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          n_features: Number of input and output features.
          n_heads: Number of attention heads in the Multi-Head Attention.
          n_hidden: Number of hidden units in the Feedforward (MLP) block.
          dropout: Dropout rate after the first layer of the MLP and the two skip connections.
        """
        super(DecoderBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(n_features)
        self.layer_norm2 = nn.LayerNorm(n_features)
        self.layer_norm3 = nn.LayerNorm(n_features)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.masked_multihead_attn = nn.MultiheadAttention(n_features, n_heads, dropout)
        self.multihead_attn = nn.MultiheadAttention(n_features, n_heads, dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features),
        )

    def forward(self, y, z, src_mask, tgt_mask):
        """
        Args:
          y of shape (max_tgt_seq_length, batch_size, n_features): Transformed target sequences used as
              the inputs of the block.
          z of shape (max_src_seq_length, batch_size, n_features): Encoded source sequences (outputs of the
              encoder).
          src_mask of shape (batch_size, max_src_seq_length): Boolean tensor indicating which elements of the
             source sequences should be ignored.
          tgt_mask of shape (max_tgt_seq_length, max_tgt_seq_length): Subsequent mask to ignore subsequent
             elements of the target sequences in the inputs. The rows of this matrix correspond to the output
             elements and the columns correspond to the input elements.

        Returns:
          z of shape (max_seq_length, batch_size, n_features): Output tensor.

        Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
        """
        attn_output, attn_output_weights = self.masked_multihead_attn(y, y, y, attn_mask=tgt_mask)
        x = self.layer_norm1(y + self.dropout_1(attn_output))
        attn_output, attn_output_weights = self.multihead_attn(x, z, z, key_padding_mask=src_mask)
        x = self.layer_norm2(x + self.dropout_2(attn_output))
        pwff = self.position_wise_feed_forward(x)
        x = self.layer_norm3(x + self.dropout_3(pwff))
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_vectors, n_blocks, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          embedding_vectors: Vector matrix from gensim pretrained vectors, accessing by model.vectors.
          n_blocks: Number of EncoderBlock blocks.
          n_heads: Number of attention heads inside the DecoderBlock.
          n_hidden: Number of hidden units in the Feedforward block of DecoderBlock.
          dropout: Dropout level used in DecoderBlock.

          # tgt_vocab_size: Number of words in the target vocabulary.
          # n_features: Number of features to be used for word embedding and further in all layers of the decoder.
        """
        super(Decoder, self).__init__()
        self.n_blocks = n_blocks
        # self.embedding = nn.Embedding(tgt_vocab_size, n_features)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_vectors))
        tgt_vocab_size = self.embedding.num_embeddings
        n_features = self.embedding.embedding_dim

        self.pos_encoding = tr.PositionalEncoding(n_features, dropout=dropout, max_len=MAX_LENGTH)
        decoder_blocks_list = []
        for i in range(self.n_blocks):
            decoder_blocks_list.append(DecoderBlock(n_features=n_features,
                                                    n_heads=n_heads,
                                                    n_hidden=n_hidden,
                                                    dropout=dropout))
        self.decoder_blocks = nn.ModuleList(decoder_blocks_list)
        self.linear = nn.Linear(n_features, tgt_vocab_size)
        self.lsm = nn.LogSoftmax(dim=2)

    def forward(self, y, z, src_mask):
        """
        Args:
          y of shape (max_tgt_seq_length, batch_size, n_features): Transformed target sequences used as
              the inputs of the block.
          z of shape (max_src_seq_length, batch_size, n_features): Encoded source sequences (outputs of the
              encoder).
          src_mask of shape (batch_size, max_src_seq_length): Boolean tensor indicating which elements of the
             source sequences should be ignored.
        Returns:
          out of shape (max_seq_length, batch_size, tgt_vocab_size): Log-softmax probabilities of the words
              in the output sequences.
        Notes:
          * All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
          * You need to create and use the subsequent mask in the decoder.
        """
        tgt_mask = subsequent_mask(y.size(0))
        x = self.pos_encoding(self.embedding(y))
        for i in range(self.n_blocks):
            x = self.decoder_blocks[i](y=x, z=z, src_mask=src_mask, tgt_mask=tgt_mask)
        out = self.lsm(self.linear(x))
        return out


def convert_src_str_to_index_seq(src_str, src_VecModel):
    src_unk_token_index = src_VecModel.vocab.get(UNK_token).index

    linesrc = preprocess_string(src_str, [strip_punctuation, strip_tags, strip_multiple_whitespaces])
    linesrc = [*linesrc, EOS_token]

    linesrc_index = []
    for w in linesrc:
        vw_index = src_VecModel.vocab.get(w)
        if vw_index is None:
            linesrc_index.append(src_unk_token_index)
        else:
            linesrc_index.append(vw_index.index)

    src_seq = torch.tensor(linesrc_index).long()
    return src_seq

def convert_tgt_index_seq_to_str(tgt_seq, tgt_VecModel):
    return


class TransformerModel():
    def __init__(self, src_vectorModel, tgt_vectorModel,
                 encoder_save_path, decoder_save_path,
                 hidden_size):
        self.encoder = None
        self.decoder = None
        self.src_vm = src_vectorModel
        self.tgt_vm = tgt_vectorModel
        self.encoder_save_path = encoder_save_path
        self.decoder_save_path = decoder_save_path
        self.hidden_size = hidden_size

    def train(self, filesrc, filetgt, batch_size=64, iters=2):
        self.encoder = Encoder(self.src_vm.vectors, n_blocks=3, n_heads=10, n_hidden=self.hidden_size)
        self.decoder = Decoder(self.tgt_vm.vectors, n_blocks=3, n_heads=10, n_hidden=self.hidden_size)

        self.encoder.to(DEVICE)
        self.decoder.to(DEVICE)

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())

        adam = torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
        optimizer = tr.NoamOptimizer(EMBEDDING_VECTOR_SIZE, 2, 10000, adam)

        trainloader = tr_data_loader(
            src_vectorModel=self.src_vm,
            tgt_vectorModel=self.tgt_vm,
            filesrc=filesrc,
            filetgt=filetgt,
            batch_size=batch_size,
            sos_token=SOS_token,
            eos_token=EOS_token,
            unk_token=UNK_token
        )

        self.encoder.train()
        self.decoder.train()

        tgt_padding_value = self.tgt_vm.vocab.get(EOS_token).index

        for epoch in range(iters):
            for i, batch in enumerate(trainloader):
                src_seqs, src_mask, tgt_seqs = batch

                src_seqs = src_seqs.to(DEVICE)
                src_mask = src_mask.to(DEVICE)
                tgt_seqs = tgt_seqs.to(DEVICE)

                src_mask = src_mask.t()
                tgt_input = tgt_seqs[:-1]
                tgt_output = tgt_seqs[1:]

                encoder_output = self.encoder(src_seqs, src_mask)
                pred = self.decoder(tgt_input, encoder_output, src_mask)

                output_to_loss = pred.view(pred.size(0) * pred.size(1), -1)
                target_to_loss = tgt_output.view(tgt_output.size(0) * tgt_output.size(1))

                loss = F.nll_loss(output_to_loss, target_to_loss, ignore_index=tgt_padding_value)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {0:d}: Loss:\t{1:0.3f}".format(epoch + 1, loss.item()))

        torch.save(self.encoder.state_dict(), self.encoder_save_path)
        torch.save(self.decoder.state_dict(), self.decoder_save_path)

    def load(self, encoder_path, decoder_path):
        self.encoder = Encoder(self.src_vm.vectors, n_blocks=3, n_heads=10, n_hidden=self.hidden_size)
        self.decoder = Decoder(self.tgt_vm.vectors, n_blocks=3, n_heads=10, n_hidden=self.hidden_size)

        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))

        self.encoder.to(DEVICE)
        self.decoder.to(DEVICE)

        self.encoder.eval()
        self.decoder.eval()


    def translate(self, src_str):
        """
        Args:
          encoder (Encoder): Trained encoder.
          decoder (Decoder): Trained decoder.
          src_seq of shape (src_seq_length): LongTensor of word indices of the source sentence.

        Returns:
          out_seq of shape (out_seq_length, 1): LongTensor of word indices of the output sentence.
        """

        src_seq = convert_src_str_to_index_seq(
            src_str=src_str,
            src_VecModel=self.src_vm)
        tgt_sos_token_index = self.tgt_vm.vocab.get(SOS_token).index

        src_seq_batch = src_seq.unsqueeze(1)
        src_mask = torch.tensor([[0] * len(src_seq)], dtype=torch.bool)

        encoder_output = self.encoder(src_seq_batch, src_mask)

        tgt_seq = torch.tensor([[tgt_sos_token_index]])
        for i in range(MAX_LENGTH - 1):
            decoder_output = self.decoder(tgt_seq, encoder_output, src_mask=src_mask)

            decoded_words = decoder_output.argmax(dim=2)
            next_word = decoded_words[-1].unsqueeze(1)
            tgt_seq = torch.cat([tgt_seq, next_word], dim=0)

        return tgt_seq



