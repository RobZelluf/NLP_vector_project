import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation, strip_tags, strip_multiple_whitespaces

import TranslationModels.tr_model_utils as tr
from TranslationModels.dataloader import tr_data_loader
from TranslationModels.const_vars import *

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import random
import numpy as np
import time

class Encoder(nn.Module):
	def __init__(self, embedding_vectors, hidden_size):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_vectors))
		self.gru = nn.GRU(input_size=self.embedding.embedding_dim, hidden_size=hidden_size)

	def forward(self, pad_seqs, seq_lengths, hidden, padding_value):
		embedded = self.embedding(pad_seqs)
		output = pack_padded_sequence(embedded, seq_lengths)
		output, hidden = self.gru(output, hidden)
		output, _ = pad_packed_sequence(output, padding_value = padding_value)
		return output, hidden

	def init_hidden(self, batch_size=1):
		return torch.zeros(1, batch_size, self.hidden_size)
	
	
	
	
class Decoder(nn.Module):
	def __init__(self, embedding_vectors, hidden_size, sos_index):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_vectors))
		self.gru = nn.GRU(input_size=self.embedding.embedding_dim, hidden_size=hidden_size)
		self.out = nn.Linear(hidden_size, self.embedding.num_embeddings)
		self.logsoft = nn.LogSoftmax(dim = 2)
		self.sos_index = sos_index

	def forward(self, hidden, pad_tgt_seqs=None, teacher_forcing=False):
		if pad_tgt_seqs is None:
			assert not teacher_forcing, 'Cannot use teacher forcing without a target sequence.'
		
		sosrow = torch.tensor([[self.sos_index] * hidden.shape[1]]).to(hidden.device)
		
		if teacher_forcing:
			embedded = self.embedding(torch.cat((sosrow, pad_tgt_seqs[:-1,:]), 0))
			output = F.relu(embedded)
			output, hidden = self.gru(output, hidden)
			output = self.out(output)
			output = self.logsoft(output)

		else:
			emb = self.embedding(sosrow)
			output = torch.empty(1, hidden.shape[1], self.embedding.num_embeddings).to(hidden.device)
			
			for i in range(MAX_LENGTH if pad_tgt_seqs is None else pad_tgt_seqs.shape[0]):
				emb = F.relu(emb)
				o, hidden = self.gru(emb, hidden)
				o = self.out(o)
				o = self.logsoft(o)
				
				o1 = torch.argmax(o, axis = 2)
				output = torch.cat([output, o], dim = 0)
				emb = self.embedding(o1)
				
			output = output[1:, :, :]

		return output, hidden


class RNNModel():
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
		try:
			self.load(self.encoder_save_path, self.decoder_save_path)
			print('++ Model loaded!')
		except:
			pass

	def train(self, filesrc, filetgt, batch_size=64, iters=2, teacher_forcing_ratio = 0.5, max_batches = None, device = "cpu"):

		src_padding_value = self.tgt_vm.vocab.get(SOS_token).index
		tgt_padding_value = self.tgt_vm.vocab.get(SOS_token).index

		if self.decoder is None:
			self.encoder = Encoder(self.src_vm.vectors, hidden_size=self.hidden_size)
		if self.decoder is None:
			self.decoder = Decoder(self.tgt_vm.vectors, hidden_size=self.hidden_size, sos_index = tgt_padding_value)

		self.encoder.to(device)
		self.decoder.to(device)

		optimizerEnc = optim.Adam(self.encoder.parameters(), lr = 0.001)
		optimizerDec = optim.Adam(self.decoder.parameters(), lr = 0.001)

		
		criterion = nn.NLLLoss(ignore_index = tgt_padding_value)

		trainloader = tr_data_loader(
			src_vectorModel=self.src_vm,
			tgt_vectorModel=self.tgt_vm,
			filesrc=filesrc,
			filetgt=filetgt,
			batch_size=batch_size,
			sos_token=SOS_token,
			eos_token=EOS_token,
			unk_token=UNK_token,
			max_batches=max_batches,
			isTransformer=False
		)

		self.encoder.train()
		self.decoder.train()

		
		start = time.time()

		for epoch in range(iters):
			for i, batch in enumerate(trainloader):
				train_inputs, train_lengths, train_targets = batch
				hidden = self.encoder.init_hidden(len(train_lengths)).to(device)
				train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
				
				optimizerEnc.zero_grad()
				optimizerDec.zero_grad()
				
				output, hidden = self.encoder(train_inputs, train_lengths, hidden, src_padding_value)
				output, hidden = self.decoder(hidden, pad_tgt_seqs = train_targets, teacher_forcing = random.random() < teacher_forcing_ratio)

				output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
				train_targets = train_targets.reshape(-1)
				
				loss = criterion(output, train_targets)

				loss.backward()
				optimizerEnc.step()
				optimizerDec.step()

			end = time.time()
			dur = end - start
			print("Epoch {0:d}: Loss:\t{1:0.3f} \t\t {0:d}m:{0:d}s".format(epoch + 1, loss.item(), dur // 60, dur % 60))

		torch.save(self.encoder.state_dict(), self.encoder_save_path)
		torch.save(self.decoder.state_dict(), self.decoder_save_path)

	def load(self, encoder_path, decoder_path):
		self.encoder = Encoder(self.src_vm.vectors, hidden_size=self.hidden_size)
		self.decoder = Decoder(self.tgt_vm.vectors, hidden_size=self.hidden_size, sos_index = self.tgt_vm.vocab.get('<SOS>').index)

		self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
		self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))

		self.encoder.to(DEVICE)
		self.decoder.to(DEVICE)

		self.encoder.eval()
		self.decoder.eval()


	def translate(self, src_str, str_out = False):

		src_padding_value = self.tgt_vm.vocab.get(SOS_token).index
		tgt_padding_value = self.tgt_vm.vocab.get(SOS_token).index

		hidden = self.encoder.init_hidden()
		
		src_seq = tr.convert_src_str_to_index_seq(
			src_str=src_str,
			src_VecModel=self.src_vm
		).unsqueeze(1)
		lens = np.array([len(src_seq),])

		output, hidden = self.encoder(src_seq, lens, hidden, src_padding_value)
		output, hidden = self.decoder(hidden, teacher_forcing = False)
		
		output = torch.argmax(output, axis = 2)
		
		try:
			output = output[:(output == EOS_token).nonzero()[0][0] + 1]
		except:
			pass

		if str_out:
			output = tr.convert_tgt_index_seq_to_str(output, self.tgt_vm)

		return output


