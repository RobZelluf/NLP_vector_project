import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import TransformerModel.tr_model_utils as tr
from TransformerModel.const_vars import MAX_LENGTH



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
    def __init__(self, src_vocab_size, n_blocks, n_features, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          src_vocab_size: Number of words in the source vocabulary.
          n_blocks: Number of EncoderBlock blocks.
          n_features: Number of features to be used for word embedding and further in all layers of the encoder.
          n_heads: Number of attention heads inside the EncoderBlock.
          n_hidden: Number of hidden units in the Feedforward block of EncoderBlock.
          dropout: Dropout level used in EncoderBlock.
        """
        super(Encoder, self).__init__()
        self.n_blocks = n_blocks
        self.embedding = nn.Embedding(src_vocab_size, n_features)
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
    def __init__(self, tgt_vocab_size, n_blocks, n_features, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          tgt_vocab_size: Number of words in the target vocabulary.
          n_blocks: Number of EncoderBlock blocks.
          n_features: Number of features to be used for word embedding and further in all layers of the decoder.
          n_heads: Number of attention heads inside the DecoderBlock.
          n_hidden: Number of hidden units in the Feedforward block of DecoderBlock.
          dropout: Dropout level used in DecoderBlock.
        """
        super(Decoder, self).__init__()
        self.n_blocks = n_blocks
        self.embedding = nn.Embedding(tgt_vocab_size, n_features)
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

