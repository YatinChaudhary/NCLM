import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import keras.preprocessing.sequence as pp
import model.utils as u
import os, sys, csv
from model.softmax import AdaptiveSoftmax
from model.softmax import FullSoftmax
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear as linear
import model.data_lstm as data

seed = 42
tf_op_seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

def max_pad_length(TM_vocab, LM_vocab, inputs, seq_lengths):
	max_len = 0
	for sent, seq_len in zip(inputs, seq_lengths):
		sent_new = []
		for counter, index in enumerate(sent[:seq_len]):
			try:
				TM_index = TM_vocab.index(LM_vocab[index])
				sent_new.append(TM_index)
			except ValueError:
				continue
		if len(sent_new) > max_len:
			max_len = len(sent_new)
	return max_len

def get_sent_topic_reps_docnade(TM_vocab, LM_vocab, inputs, seq_lengths, pad_length=None, pad_value=0):
	inputs_new = np.ones((inputs.shape[0], inputs.shape[1], pad_length)).astype(np.int32) * pad_value
	for ii, (sent, seq_len) in enumerate(zip(inputs, seq_lengths)):
		for jj in range(1, seq_len):
			counter = 0
			for index_id, index in enumerate(sent[:seq_len]):
				if (index_id != jj):
					try:
						TM_index = TM_vocab.index(LM_vocab[index])
						inputs_new[ii, jj, counter] = TM_index
						counter += 1
					except ValueError:
						continue
	return np.array(inputs_new, dtype=np.int32)

def get_sent_topic_reps_nvdm(TM_vocab, LM_vocab, inputs, seq_lengths):
	inputs_new = np.zeros((inputs.shape[0], inputs.shape[1], len(TM_vocab))).astype(np.float32)
	for ii, (sent, seq_len) in enumerate(zip(inputs, seq_lengths)):
		for jj in range(1, seq_len):
			for index_id, index in enumerate(sent[:seq_len]):
				if (index_id != jj):
					try:
						TM_index = TM_vocab.index(LM_vocab[index])
						inputs_new[ii, jj, TM_index] += 1
					except ValueError:
						continue
	return np.array(inputs_new, dtype=np.float32)

def get_doc_topic_mask_nvdm(TM_vocab, LM_vocab, inputs_TM, inputs_LM):
	inputs_new = np.zeros((inputs_TM.shape[0], len(TM_vocab))).astype(np.float32)
	for ii, (doc_minus_sent, sent) in enumerate(zip(inputs_TM, inputs_LM)):
		for TM_index in doc_minus_sent.nonzero()[0]:
			inputs_new[ii, TM_index] = 1
		
		for index in sent:
			try:
				LM_index = TM_vocab.index(LM_vocab[index])
				inputs_new[ii, LM_index] = 1
			except ValueError:
				continue
	return np.array(inputs_new, dtype=np.float32)

def get_char_indices(rnn_input, vocab_word, vocab_char):
	max_length_word = max([max(map(lambda x: len(vocab_word[x].strip()), seq)) for seq in rnn_input])

	rnn_char_input = []
	rnn_char_sequence_lengths = []
	for doc in rnn_input:
		temp_doc = []
		temp_sequence_lengths = []
		for index in doc:
			char_ids = []
			word = vocab_word[int(index)]
			word_length = len(word)
			temp_sequence_lengths.append(word_length)
			for char in list(word):
				char_index = vocab_char.index(char)
				char_ids.append(char_index)
			for pad_index in range(max_length_word - word_length):
				char_ids.append(0)
			temp_doc.append(char_ids)
		rnn_char_input.append(temp_doc)
		rnn_char_sequence_lengths.append(temp_sequence_lengths)
	
	return rnn_char_input, rnn_char_sequence_lengths

class BiLstmCrf(object):
	def __init__(self, params, 
				TM_loss_weight=None,
				LM_loss_weight=None,
				topic_model=None,
				activation=None, 
				W_initializer=None):

		self.TM_loss_weight = TM_loss_weight
		self.LM_loss_weight = LM_loss_weight

		#params.lstm_dropout_keep_prob = 0.6
		#params.hidden_size_LM = 200
		
		self.x_rnn_input = tf.placeholder(tf.int32, shape=(None, None), name='x_rnn_input')
		self.x_rnn_output = tf.placeholder(tf.int32, shape=(None, None), name='x_rnn_output')
		if params.multi_label:
			self.y_rnn = tf.placeholder(tf.string, shape=(None), name='y_rnn')
		else:
			self.y_rnn = tf.placeholder(tf.int32, shape=(None), name='y_rnn')
		self.rnn_seq_lengths = tf.placeholder(tf.int32, shape=(None), name='rnn_seq_lengths')
		self.lstm_dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name='lstm_dropout_keep_prob')
		self.tm_dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name='tm_dropout_keep_prob')

		self.topic_model = topic_model
		batch_size = tf.shape(self.x_rnn_input)[0]
		batch_length = tf.shape(self.x_rnn_input)[1]

		if params.use_char_embeddings:
			self.x_rnn_char_input = tf.placeholder(tf.int32, shape=(None, None, None), name='x_char_rnn')
			self.rnn_char_seq_lengths = tf.placeholder(tf.int32, shape=(None, None), name='rnn_char_seq_lengths')

		rnn_seq_lengths_input = self.rnn_seq_lengths

		if params.use_topic_embedding:
			"""
			with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
				W_proj_initializer = tf.glorot_uniform_initializer(seed=tf_op_seed)
				W_proj = tf.get_variable(
						'Topic_embedding_projection_matrix_rnn',
						[params.TM_vocab_length, params.hidden_size_LM],
						dtype=tf.float32,
						initializer=W_proj_initializer,
						trainable=True
					)
			"""

		if params.combined_training:
			if params.use_topic_embedding:
				#topic_model_h = tf.matmul(self.topic_model.last_h_topic_emb, W_proj, name='TM_doc_topic_emb_proj')
				topic_model_h = self.topic_model.last_h_topic_emb
				#topic_model_h = tf.nn.dropout(topic_model_h, self.tm_dropout_keep_prob, seed=seed)
				topic_model_h = tf.expand_dims(topic_model_h, axis=1, name='TM_doc_topic_emb_expand')
				topic_model_h = tf.tile(topic_model_h, [1, batch_length, 1], name='TM_doc_topic_emb_tile')
				#topic_vector_size = params.hidden_size_LM
				topic_vector_size = self.topic_model.topic_emb_size
				if params.use_sent_topic_rep:
					#topic_model_h_sent = tf.matmul(self.topic_model.last_h_topic_emb_sent, W_proj, name='TM_sent_topic_emb_proj')
					topic_model_h_sent = self.topic_model.last_h_topic_emb_sent
					#topic_model_h_sent = tf.reshape(topic_model_h_sent, [batch_size, -1, topic_vector_size], name='TM_sent_topic_emb_proj_reshape')
					if not params.use_bilm:
						topic_model_h_sent = topic_model_h_sent[:, 1:, :]
					topic_model_h = tf.concat([topic_model_h, topic_model_h_sent], axis=2, name='doc_topic_emb_concat_sent')
					topic_vector_size = topic_vector_size * 2
				#topic_vector_size = params.TM_vocab_length
			else:
				topic_model_h = self.topic_model.last_h
				#topic_model_h = tf.nn.dropout(topic_model_h, self.tm_dropout_keep_prob, seed=seed)
				topic_model_h = tf.expand_dims(topic_model_h, axis=1, name='TM_h_expand')
				topic_model_h = tf.tile(topic_model_h, [1, batch_length, 1], name='TM_h_tile')

				if params.TM_type == "docnade":
					topic_vector_size = params.hidden_size_TM
				elif params.TM_type == "nvdm":
					topic_vector_size = self.topic_model.n_topic
				else:
					print("Error: wrong value of param.TM_type %s" % params.TM_type)
					sys.exit()

				if params.use_sent_topic_rep:
					topic_model_h_sent = self.topic_model.last_h_sent
					if not params.use_bilm:
						topic_model_h_sent = topic_model_h_sent[:, 1:, :]
					topic_model_h = tf.concat([topic_model_h, topic_model_h_sent], axis=2, name='doc_topic_prop_concat_sent')
					topic_vector_size = topic_vector_size * 2
					#topic_model_h = topic_model_h_sent
					#topic_vector_size = topic_vector_size
		
		if params.use_bilm:
			topic_model_h = topic_model_h[:, 1:-1, :]
		#else:
		#	topic_model_h = topic_model_h[:, 1:, :]
		
		# Do an embedding lookup for each word in each sequence
		with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
			# Initialisation scheme taken from the original DocNADE source
			if W_initializer is None:
				W_initializer = tf.glorot_uniform_initializer(seed=tf_op_seed)

				W = tf.get_variable(
					'word_embeddings_rnn',
					#[params.LM_vocab_length, params.hidden_size_LM],
					[params.LM_vocab_length, params.input_size_LM],
					dtype=tf.float32,
					initializer=W_initializer,
					trainable=True
				)
			else:
				W = tf.get_variable(
					'word_embeddings_rnn',
					dtype=tf.float32,
					initializer=W_initializer,
					trainable=True
				)

			if params.use_MOR and params.combined_training and (not params.use_topic_embedding):
				print("Error: 1: this section will not work if (Doc-sent) and (sent-word) topic proportions are concatenated together.")
				sys.exit()

				#W_tile = tf.tile(tf.expand_dims(W, axis=0), [params.hidden_size_TM, 1, 1], name='W_tile')
				#W_tile_reshape = tf.reshape(W_tile, [params.hidden_size_TM, -1], name='W_tile_reshape')
				W_tile_reshape = tf.reshape(W, [params.hidden_size_TM, params.LM_vocab_length * params.hidden_size_LM], name='W_tile_reshape')
				W_topic_weights = tf.nn.softmax(topic_model_h, axis=1, name='W_topic_weights')
				W_tile_reshape_dot = tf.keras.backend.dot(W_topic_weights, W_tile_reshape, name='W_tile_reshape_dot')
				W_final = tf.reshape(W_tile_reshape_dot, [params.hidden_size_TM, params.LM_vocab_length, params.hidden_size_LM], name='W_final')

				elems = (W_final, self.x_rnn_input)
				self.embeddings = tf.map_fn(
					lambda inputs: tf.nn.embedding_lookup(inputs[0], inputs[1]), 
					elems, 
					dtype=tf.float32,
					name='rnn_mor_word_embedding_lookup'
				)
			else:
				self.embeddings = tf.nn.embedding_lookup(W, self.x_rnn_input, name='rnn_word_embedding_lookup')
				self.embeddings = tf.nn.dropout(self.embeddings, keep_prob=self.lstm_dropout_keep_prob, seed=seed)

		lstm_cell_initializer = tf.glorot_uniform_initializer(seed=tf_op_seed)
		if params.use_char_embeddings:
			with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
				# Initialisation scheme taken from the original DocNADE source
				W_char_initializer = tf.glorot_uniform_initializer(seed=tf_op_seed)

				W_char = tf.get_variable(
					'char_embeddings_rnn',
					[params.rnn_char_vocab_length, params.hidden_size_LM_char],
					dtype=tf.float32,
					initializer=W_char_initializer,
					trainable=True
				)

				self.char_embeddings = tf.nn.embedding_lookup(W_char, x_rnn_char_input, name='rnn_char_embedding_lookup')

				# put the time dimension on axis=1
				s = tf.shape(self.char_embeddings)
				self.char_embeddings = tf.reshape(self.char_embeddings,
						shape=[s[0]*s[1], s[-2], params.hidden_size_LM_char])
				word_lengths = tf.reshape(rnn_char_seq_lengths, shape=[s[0]*s[1]])

				# bi lstm on chars
				cell_fw = rnn.LSTMCell(params.hidden_size_LM_char, forget_bias=1.0, 
						initializer=lstm_cell_initializer, state_is_tuple=True, name='char_lstm_cell_fw')
				cell_bw = rnn.LSTMCell(params.hidden_size_LM_char, forget_bias=1.0, 
						initializer=lstm_cell_initializer, state_is_tuple=True, name='char_lstm_cell_bw')
				char_output = tf.nn.bidirectional_dynamic_rnn(
						cell_fw, cell_bw, self.char_embeddings,
						sequence_length=word_lengths, dtype=tf.float32)

				# read and concat output
				_, ((_, char_output_fw), (_, char_output_bw)) = char_output
				char_output = tf.concat([char_output_fw, char_output_bw], axis=-1)
				self.char_output = char_output

				# shape = (batch size, max sentence length, char hidden size)
				char_output = tf.reshape(char_output,
						shape=[s[0], s[1], 2 * params.hidden_size_LM_char])
				self.char_output_reshape = char_output
				self.embeddings = tf.concat([self.embeddings, char_output], axis=-1)
		
		if not params.deep:
			with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
				if params.use_bilm:
					#lstm_cell_fw = rnn.LSTMCell(params.hidden_size_LM, activation=activation, forget_bias=1.0, name='lstm_cell_fw')
					lstm_cell_fw = rnn.LSTMCell(params.hidden_size_LM, forget_bias=1.0, initializer=lstm_cell_initializer, name='lstm_cell_fw')
					#lstm_cell_bw = rnn.LSTMCell(params.hidden_size_LM, activation=activation, forget_bias=1.0, name='lstm_cell_bw')
					lstm_cell_bw = rnn.LSTMCell(params.hidden_size_LM, forget_bias=1.0, initializer=lstm_cell_initializer, name='lstm_cell_bw')
					#set initial state to all zeros
					initial_state_fw = lstm_cell_fw.zero_state(batch_size, tf.float32)
					initial_state_bw = lstm_cell_bw.zero_state(batch_size, tf.float32)
					if params.lstm_dropout_keep_prob >= 0.0:
						lstm_cell_fw = rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=self.lstm_dropout_keep_prob, seed=seed)
						lstm_cell_bw = rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=self.lstm_dropout_keep_prob, seed=seed)
					
					(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
							lstm_cell_fw, lstm_cell_bw, self.embeddings,
							#sequence_length=self.rnn_seq_lengths, initial_state_fw=initial_state_fw, 
							sequence_length=rnn_seq_lengths_input, initial_state_fw=initial_state_fw, 
							initial_state_bw=initial_state_bw, dtype=tf.float32)
					output_fw = output_fw[:, :-2, :]
					output_bw = output_bw[:, 2:, :]
					outputs = tf.concat([output_fw, output_bw], axis=2)
					params.hidden_size_LM *= 2
					batch_length -= 2
				else:
					#lstm_cell = rnn.LSTMCell(params.hidden_size_LM, activation=activation, forget_bias=1.0, name='lstm_cell')
					lstm_cell = rnn.LSTMCell(params.hidden_size_LM, forget_bias=1.0, initializer=lstm_cell_initializer, name='lstm_cell')
					#set initial state to all zeros
					initial_state = lstm_cell.zero_state(batch_size, tf.float32)
					if params.lstm_dropout_keep_prob >= 0.0:
						lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.lstm_dropout_keep_prob, seed=seed)
					outputs, state = tf.nn.dynamic_rnn(lstm_cell, self.embeddings, 
									#sequence_length=self.rnn_seq_lengths, initial_state=initial_state, dtype=tf.float32)
									sequence_length=rnn_seq_lengths_input, initial_state=initial_state, dtype=tf.float32)
				#outputs = tf.nn.dropout(outputs, rate=1-self.lstm_dropout_keep_prob)
		else:
			with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
				if params.use_bilm:
					lstm_cells_fw = []
					lstm_cells_bw = []
					initial_states_fw = []
					initial_states_bw = []
					for i, size in enumerate(params.deep_hidden_sizes):
						#lstm_cell_fw = rnn.LSTMCell(params.hidden_size_LM, activation=activation, forget_bias=1.0, name='lstm_cell_fw')
						lstm_cell_fw = rnn.LSTMCell(params.hidden_size_LM, forget_bias=1.0, initializer=lstm_cell_initializer, name='lstm_cell_fw')
						#lstm_cell_bw = rnn.LSTMCell(params.hidden_size_LM, activation=activation, forget_bias=1.0, name='lstm_cell_bw')
						lstm_cell_bw = rnn.LSTMCell(params.hidden_size_LM, forget_bias=1.0, initializer=lstm_cell_initializer, name='lstm_cell_bw')
						#set initial state to all zeros
						initial_state_fw = lstm_cell_fw.zero_state(batch_size, tf.float32)
						initial_state_bw = lstm_cell_bw.zero_state(batch_size, tf.float32)

						if params.lstm_dropout_keep_prob >= 0.0:
							lstm_cell_fw = rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=self.lstm_dropout_keep_prob, seed=seed)
							lstm_cell_bw = rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=self.lstm_dropout_keep_prob, seed=seed)

						lstm_cells_fw.append(lstm_cell_fw)
						lstm_cells_bw.append(lstm_cell_bw)
						initial_states_fw.append(initial_state_fw)
						initial_states_bw.append(initial_state_bw)
					
					outputs_, output_state_fw, output_state_bw = rnn.stack_bidirectional_dynamic_rnn(lstm_cells_fw, lstm_cells_bw, 
																self.embeddings, initial_states_fw=initial_states_fw,
																initial_states_bw=initial_states_bw, dtype=tf.float32,
																#sequence_length=self.rnn_seq_lengths)
																sequence_length=rnn_seq_lengths_input)
					output_fw, output_bw = tf.split(outputs_, 2, axis=2)
					output_fw = output_fw[:, :-2, :]
					output_bw = output_bw[:, 2:, :]
					outputs = tf.concat([output_fw, output_bw], axis=2)
					params.hidden_size_LM *= 2
					batch_length -= 2
				else:
					lstm_cells = []
					initial_states = []
					for i, size in enumerate(params.deep_hidden_sizes):
						#lstm_cell = rnn.LSTMCell(size, activation=activation, forget_bias=1.0, name='lstm_cell' + str(i))
						lstm_cell = rnn.LSTMCell(size, forget_bias=1.0, initializer=lstm_cell_initializer, name='lstm_cell' + str(i))
						#set initial state to all zeros
						initial_state = lstm_cell.zero_state(batch_size, tf.float32)
						if params.lstm_dropout_keep_prob:
							lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.lstm_dropout_keep_prob, seed=seed)
						lstm_cells.append(lstm_cell)
						initial_states.append(initial_state)
					stacked_lstm_cell = rnn.MultiRNNCell(cells=lstm_cells)
					initial_state = stacked_lstm_cell.zero_state(batch_size, tf.float32)
					#outputs, state = tf.nn.dynamic_rnn(stacked_lstm_cell, self.embeddings, sequence_length=self.rnn_seq_lengths, 
					outputs, state = tf.nn.dynamic_rnn(stacked_lstm_cell, self.embeddings, sequence_length=rnn_seq_lengths_input, 
									#initial_state=initial_states, dtype=tf.float32)
									initial_state=(initial_states[0], initial_states[1]), dtype=tf.float32)

		self.outputs = outputs
		self.outputs_shape = tf.shape(outputs)

		self.indices = tf.stack([
			tf.range(batch_size),
			tf.to_int32(rnn_seq_lengths_input) - 1
		], axis=1)
		self.last_lstm_h = tf.gather_nd(self.outputs, self.indices, name="last_hidden_lstm")
		self.outputs = tf.reshape(self.outputs, [-1, params.hidden_size_LM])
		
		if params.combined_training:
			if params.combination_type == "concat":
				combined_dim = topic_vector_size + params.hidden_size_LM
				projected_dim = params.hidden_size_LM
				#projected_dim = 750
				
				outputs = tf.concat([outputs, topic_model_h], axis=2, name='combined_h')
				self.combined_h = outputs
				self.last_comb_h = tf.gather_nd(self.combined_h, self.indices, name="last_hidden_comb")

				outputs = tf.reshape(outputs, [-1, combined_dim], name='combined_h_reshape')

				if params.common_space:
					if params.concat_proj_activation == 'sigmoid':
						proj_non_lin = tf.sigmoid
					elif params.concat_proj_activation == 'tanh':
						proj_non_lin = tf.tanh
					elif params.concat_proj_activation == 'relu':
						proj_non_lin = tf.nn.relu
					elif params.concat_proj_activation == 'linear':
						proj_non_lin = tf.keras.activations.linear
					else:
						print('Invalid value for activation: %s' % (params.concat_proj_activation))
						exit()
					
					with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
						W_proj_rnn_initializer = tf.glorot_uniform_initializer(seed=tf_op_seed)
						W_proj_rnn = tf.get_variable(
							'W_proj_rnn',
							#[combined_dim, params.hidden_size_LM],
							[combined_dim, projected_dim],
							dtype=tf.float32,
							initializer=W_proj_rnn_initializer,
							trainable=True
						)

						bias_proj_rnn = tf.get_variable(
							'bias_proj_rnn',
							#[params.hidden_size_LM],
							[projected_dim],
							dtype=tf.float32,
							initializer=tf.constant_initializer(0.0),
							trainable=True
						)
					
					#outputs = tf.nn.xw_plus_b(outputs, W_proj_rnn, bias_proj_rnn, name='2d_logits_lstm')
					outputs = proj_non_lin(tf.nn.xw_plus_b(outputs, W_proj_rnn, bias_proj_rnn, name='2d_logits_lstm'))
					#self.combined_h_proj = tf.reshape(outputs, [batch_size, -1, params.hidden_size_LM], name='combined_h_proj')
					self.combined_h_proj = tf.reshape(outputs, [batch_size, -1, projected_dim], name='combined_h_proj')
					self.last_comb_h_proj = tf.gather_nd(self.combined_h_proj, self.indices, name="last_hidden_comb_proj")
				else:
					self.last_comb_h_proj = self.last_comb_h
			elif params.combination_type == "sum":
				print("Error: 2: this section will not work if (Doc-sent) and (sent-word) topic proportions are concatenated together.")
				sys.exit()
				
				combined_dim = params.hidden_size_LM
				outputs = tf.add(outputs, tf.scalar_mul(params.TM_lambda, topic_model_h), name='combined_h')
				self.combined_h = outputs
				self.last_comb_h = tf.gather_nd(self.combined_h, self.indices, name='last_hidden_comb')
				self.last_comb_h_proj = self.last_comb_h
				outputs = tf.reshape(outputs, [-1, params.hidden_size_LM])
			elif params.combination_type == "gating":
				combined_dim = params.hidden_size_LM

				with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
					gate_initializer = tf.glorot_uniform_initializer(seed=tf_op_seed)
					self.gate_w = tf.get_variable("gate_w", [topic_vector_size, params.hidden_size_LM], \
									initializer = gate_initializer)
					self.gate_u = tf.get_variable("gate_u", [params.hidden_size_LM, params.hidden_size_LM], \
									initializer = gate_initializer)
					self.gate_b = tf.get_variable("gate_b", [params.hidden_size_LM], \
									initializer=tf.constant_initializer(0.0))
				
				topic_model_h = tf.reshape(topic_model_h, [-1, topic_vector_size], name='topic_model_h_reshape')
				outputs = tf.reshape(outputs, [-1, params.hidden_size_LM], name='outputs_reshape')

				#combine topic and language model hidden units with a gating unit
				linear_mat_init = tf.glorot_uniform_initializer(seed=tf_op_seed)
				linear_bias_init = tf.constant_initializer(1.0)
				z, r = tf.split(linear(
									[topic_model_h, outputs],
									2 * params.hidden_size_LM,
									True,
									bias_initializer=linear_bias_init,
									kernel_initializer=linear_mat_init
								),
							num_or_size_splits=2, axis=1, name='split_op'
							#num_or_size_splits=[topic_vector_size, params.hidden_size_LM], axis=1, name='split_op'
						)
				z, r = tf.sigmoid(z, name='z_sigmoid'), tf.sigmoid(r, name='r_sigmoid')
				c = tf.tanh(tf.matmul(topic_model_h, self.gate_w) + tf.matmul(tf.multiply(r, outputs), self.gate_u) + \
					self.gate_b)
				outputs = tf.multiply((1-z), outputs) + tf.multiply(z, c)

				self.combined_h = tf.reshape(outputs, [batch_size, batch_length, params.hidden_size_LM], name='combined_h')
				self.last_comb_h = tf.gather_nd(self.combined_h, self.indices, name="last_hidden_comb")
				self.last_comb_h_proj = self.last_comb_h
			else:
				print("Wrong value for params.combination_type: ", params.combination_type)
				sys.exit()
		else:
			self.last_comb_h = self.last_lstm_h
			self.last_comb_h_proj = self.last_lstm_h
			outputs = tf.reshape(outputs, [-1, params.hidden_size_LM])

		if params.use_bilm:
			#rnn_seq_lengths = tf.to_int32(self.rnn_seq_lengths) - 2
			rnn_seq_lengths_output = self.rnn_seq_lengths - 2
		else:
			rnn_seq_lengths_output = self.rnn_seq_lengths
		
		## Softmax option
		if params.softmax_type == "adaptive":
			self.softmax_layer_pretrain = AdaptiveSoftmax(params.hidden_size_LM, [2000, params.LM_vocab_length], scope='softmax_pretrain_RNN')
			if params.combination_type == 'concat':
				if params.common_space:
					self.softmax_layer_comb = self.softmax_layer_pretrain
				else:
					self.softmax_layer_comb = AdaptiveSoftmax(combined_dim, [2000, params.LM_vocab_length], scope='softmax_combined_RNN')
			elif (params.combination_type == 'sum') or (params.combination_type == 'gating'):
				#self.softmax_layer_comb = AdaptiveSoftmax(combined_dim, [2000, params.LM_vocab_length], scope='softmax_pretrain_RNN')
				self.softmax_layer_comb = self.softmax_layer_pretrain
		elif params.softmax_type == "full":
			self.softmax_layer_pretrain = FullSoftmax(params.hidden_size_LM, params.LM_vocab_length, scope='softmax_pretrain_RNN')
			if params.combination_type == 'concat':
				if params.common_space:
					if projected_dim == params.hidden_size_LM:
						self.softmax_layer_comb = self.softmax_layer_pretrain
					else:
						self.softmax_layer_comb = FullSoftmax(projected_dim, params.LM_vocab_length, scope='softmax_combined_RNN')
				else:
					#self.softmax_layer_comb = FullSoftmax(combined_dim, params.LM_vocab_length, scope='softmax_combined_RNN')
					self.softmax_layer_comb = FullSoftmax(combined_dim, params.LM_vocab_length, scope='softmax_combined_RNN',
															concat_V=self.softmax_layer_pretrain.softmax_w,
															concat_dim=params.hidden_size_LM)
			elif (params.combination_type == 'sum') or (params.combination_type == 'gating'):
				#self.softmax_layer_comb = FullSoftmax(combined_dim, params.LM_vocab_length, scope='softmax_pretrain_RNN')
				self.softmax_layer_comb = self.softmax_layer_pretrain
		else:
			print("Error: wrong value for params.softmax_type: %s" % params.softmax_type)
			sys.exit()
		
		if params.use_crf:
			## Adaptive softmax is not added
			self.logits_comb = u.linear(outputs, params.LM_vocab_length, scope='softmax_combined_RNN', suffix="_lstm")
			self.logits_comb = tf.reshape(self.logits_comb, [-1, batch_length, params.LM_vocab_length])
			log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
					#self.logits_comb, self.x_rnn_output, self.rnn_seq_lengths)
					self.logits_comb, self.x_rnn_output, rnn_seq_lengths_output)
			self.trans_params = trans_params # need to evaluate it for decoding
			self.total_loss = tf.reduce_mean(-log_likelihood)

			if params.pretrain_LM:
				self.logits_pretrain = u.linear(self.outputs, params.LM_vocab_length, scope='softmax_pretrain_RNN', suffix="_lstm_pretrain")
				self.logits_pretrain = tf.reshape(self.logits_pretrain, [-1, batch_length, params.LM_vocab_length])
				log_likelihood_pretrain, trans_params_pretrain = tf.contrib.crf.crf_log_likelihood(
						#self.logits_pretrain, self.x_rnn_output, self.rnn_seq_lengths)
						self.logits_pretrain, self.x_rnn_output, rnn_seq_lengths_output)
				self.trans_params_pretrain = trans_params_pretrain # need to evaluate it for decoding
				self.total_loss_pretrain = tf.reduce_mean(-log_likelihood_pretrain)
		else:
			if not params.num_samples:
				self.logits_comb = self.softmax_layer_comb.logits(outputs)
				if params.pretrain_LM:
					self.logits_pretrain = self.softmax_layer_pretrain.logits(self.outputs)
				loss_function = None
			else:
				## Take code from previous version file
				print("Error: wrong choice for params.num_samples %" % params.num_samples)
				sys.exit()
			
			self.loss_normed_comb, _, self.mask_comb, self.loss_unnormed_comb, self.training_losses_comb  = self.softmax_layer_comb.loss(
				self.x_rnn_output,
				#self.rnn_seq_lengths,
				rnn_seq_lengths_output,
				outputs,
				loss_function=loss_function,
				norm_by_seq_lengths=True,
				name='TM_LM_loss'
			)

			if params.pretrain_LM:
				self.loss_normed_lstm, _, self.mask_lstm, self.loss_unnormed_lstm, self.training_losses_lstm  = self.softmax_layer_pretrain.loss(
					self.x_rnn_output,
					#self.rnn_seq_lengths,
					rnn_seq_lengths_output,
					self.outputs,
					loss_function=loss_function,
					norm_by_seq_lengths=True,
					name='LM_loss'
				)

				self.total_loss_pretrain = self.training_losses_lstm

		if params.pretrain_LM:
			rnn_pretrain_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RNN')
			rnn_softmax_pretrain_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softmax_pretrain_RNN')

			# Optimiser
			step_lstm = tf.Variable(0, trainable=False)
			optimizer_pretrain_lstm = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
			self.opt_pretrain_lstm = u.gradients(
				opt=optimizer_pretrain_lstm,
				loss=self.total_loss_pretrain,
				vars=rnn_pretrain_trainable_variables + rnn_softmax_pretrain_trainable_variables,
				step=step_lstm,
				name="pretrain_LM_optimizer"
			)

		if params.combined_training:
			#self.total_loss = []
			#self.total_loss.extend(self.LM_loss_weight * self.training_losses_comb)
			#self.total_loss.extend(self.TM_loss_weight * self.topic_model.objective_TM)
			self.total_loss = self.TM_loss_weight * self.topic_model.objective_TM + self.LM_loss_weight * self.training_losses_comb
			#self.total_loss = self.TM_loss_weight * tf.reduce_mean(self.topic_model.objective_TM) + self.LM_loss_weight * tf.reduce_mean(self.loss_unnormed_comb)

		if params.combined_training:
			if params.TM_type == "docnade":
				TM_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM') + \
										tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softmax_pretrain_TM')
			elif params.TM_type == "nvdm":
				TM_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM_encoder') + \
										tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM_decoder')
			rnn_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RNN')
			if params.combination_type == 'concat':
				if params.common_space:
					if projected_dim == params.hidden_size_LM:
						rnn_softmax_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softmax_pretrain_RNN')
					else:
						rnn_softmax_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softmax_combined_RNN')
				else:
					rnn_softmax_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softmax_combined_RNN')
			elif (params.combination_type == 'sum') or (params.combination_type == 'gating'):
				rnn_softmax_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softmax_pretrain_RNN')
			else:
				print("Wrong value for params.combination_type: ", params.combination_type)
				sys.exit()

			# Optimizer
			step = tf.Variable(0, trainable=False)
			optimizer_comb = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
			self.opt_comb = u.gradients(
				opt=optimizer_comb,
				loss=self.total_loss,
				vars=TM_trainable_variables + rnn_trainable_variables + rnn_softmax_trainable_variables,
				step=step,
				name="TM_LM_optimizer"
			)
	
	## Pretraining of LSTM-LM

	def pretrain(self, dataset, params, session):
		log_dir = os.path.join(params.model, 'logs_lstm_pretrain')
		model_dir_ir_lstm = os.path.join(params.model, 'model_ir_lstm_pretrain')
		model_dir_ppl_lstm = os.path.join(params.model, 'model_ppl_lstm_pretrain')
		model_dir_supervised_lstm = os.path.join(params.model, 'model_supervised_lstm_pretrain')

		if not os.path.isdir(log_dir):
			os.mkdir(log_dir)
		if not os.path.isdir(model_dir_ir_lstm):
			os.mkdir(model_dir_ir_lstm)
		if not os.path.isdir(model_dir_ppl_lstm):
			os.mkdir(model_dir_ppl_lstm)
		if not os.path.isdir(model_dir_supervised_lstm):
			os.mkdir(model_dir_supervised_lstm)

		avg_loss = tf.placeholder(tf.float32, [], 'loss_ph')
		tf.summary.scalar('loss', avg_loss)

		validation = tf.placeholder(tf.float32, [], 'validation_ph')
		validation_accuracy = tf.placeholder(tf.float32, [], 'validation_acc')
		tf.summary.scalar('validation', validation)
		tf.summary.scalar('validation_accuracy', validation_accuracy)

		summary_writer = tf.summary.FileWriter(log_dir, session.graph)
		summaries = tf.summary.merge_all()

		rnn_pretrain_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RNN')
		rnn_softmax_pretrain_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softmax_pretrain_RNN')
		pretrain_saver = tf.train.Saver(
			rnn_pretrain_trainable_variables + \
			rnn_softmax_pretrain_trainable_variables
		)

		losses = []

		# This currently streams from disk. You set num_epochs=1 and
		# wrap this call with something like itertools.cycle to keep
		# this data in memory.

		training_data_rnn = dataset.batches_split('training_lstm_sents', params.batch_size, shuffle=False, multilabel=params.multi_label)

		num_train_docs = 0
		for y, _ in dataset.rows('training_lstm_sents', num_epochs=1):
			num_train_docs += 1

		pretrain_num_steps = int(np.round((params.pretrain_epochs * num_train_docs) / params.batch_size))
		#pretrain_num_steps = int(np.round((10 * num_train_docs) / params.batch_size))
		
		best_val_lstm_IR = -1.0
		best_val_lstm_ppl = np.inf
		best_val_lstm_nll = np.inf
		best_val_disc_accuracy = -1.0

		best_test_lstm_IR = -1.0
		best_test_lstm_ppl = np.inf
		best_test_lstm_nll = np.inf
		best_test_disc_accuracy = -1.0
		
		patience_count = 0
		best_train_nll = np.inf

		ppl_model = False
		ir_model = False

		training_labels = np.array(
			[[y] for y, _ in dataset.rows('training_lstm_docs', num_epochs=1)]
		)
		validation_labels = np.array(
			[[y] for y, _ in dataset.rows('validation_lstm_docs', num_epochs=1)]
		)
		test_labels = np.array(
			[[y] for y, _ in dataset.rows('test_lstm_docs', num_epochs=1)]
		)

		if params.use_char_embeddings:
			with open(params.rnnVocab, 'r') as f:
				vocab_rnn_word = [w.strip() for w in f.readlines()]

			with open(params.rnnCharVocab, 'r') as f:
				vocab_rnn_char = [w.strip() for w in f.readlines()]
		
		for step in range(pretrain_num_steps + 1):
			this_loss = -1.

			y_rnn, x_rnn, rnn_seq_lengths, _, _, _ = next(training_data_rnn)
			if params.use_bilm:
				x_rnn_input = x_rnn
				x_rnn_output = x_rnn[:, 1:-1]
			else:
				x_rnn_input = x_rnn[:, :-1]
				x_rnn_output = x_rnn[:, 1:]
				rnn_seq_lengths -= 1

			train_feed_dict = {
				self.x_rnn_input: x_rnn_input,
				self.x_rnn_output: x_rnn_output,
				self.y_rnn: y_rnn,
				self.rnn_seq_lengths: rnn_seq_lengths,
				self.lstm_dropout_keep_prob: params.lstm_dropout_keep_prob
			}

			if params.use_char_embeddings:
				x_rnn_char_input, rnn_char_seq_lengths = get_char_indices(x_rnn_input, vocab_rnn_word, vocab_rnn_char)
				train_feed_dict[self.x_rnn_char_input] = x_rnn_char_input
				train_feed_dict[self.rnn_char_seq_lengths] = rnn_char_seq_lengths
			
			_, loss_unnormed = session.run([self.opt_pretrain_lstm, self.loss_unnormed_lstm], feed_dict=train_feed_dict)

			loss_unnormed = np.mean(loss_unnormed)
			
			losses.append(loss_unnormed)

			if (step % params.log_every == 0):
				print('{}: {:.6f}'.format(step, loss_unnormed))	
			

			#if step >= 1 and step % params.validation_ppl_freq == 0:
			if step >= 1 and step % params.lstm_validation_ppl_freq == 0:
				ppl_model = True

				total_val_nll, total_val_ppl = self.run_epoch(
					dataset.batches_split('validation_lstm_sents', params.batch_size, num_epochs=1, shuffle=False, multilabel=params.multi_label),
					params,
					session
				)

				# Saving model and Early stopping on PPL
				if total_val_ppl < best_val_lstm_ppl:
					best_val_lstm_ppl = total_val_ppl
					print('saving: {}'.format(model_dir_ppl_lstm))
					pretrain_saver.save(session, model_dir_ppl_lstm + '/model_ppl_lstm_pretrain', global_step=1)
					patience_count = 0
				else:
					patience_count += 1
				
				if total_val_nll < best_val_lstm_nll:
					best_val_lstm_nll = total_val_nll

				print('LSTM val PPL: {:.3f}	(Best LSTM val PPL: {:.3f}),	LSTM val NLL: {:.3f}	(Best LSTM val NLL: {:.3f})'.format(
					total_val_ppl,
					best_val_lstm_ppl,
					total_val_nll,
					best_val_lstm_nll
				))

				# logging information
				with open(os.path.join(log_dir, "training_info_ppl_pretrain.txt"), "a") as f:
					f.write("Step: %i,	LSTM val PPL: %s	(Best LSTM val PPL: %s),	LSTM val NLL: %s	(Best val NLL: %s)\n" %
							(step, total_val_ppl, best_val_lstm_ppl, total_val_nll, best_val_lstm_nll))

				if (step / params.lstm_validation_ppl_freq) == 10:
					pretrain_saver.save(session, os.path.join(params.model, 'model_ppl_lstm_pretrain_10th_epoch') + '/model_ppl_lstm_pretrain', global_step=1)
					with open(os.path.join(log_dir, "training_info_ppl_pretrain.txt"), "a") as f:
						f.write("Saving model at 10th epoch.\n")

				if (step / params.lstm_validation_ppl_freq) == 11:
					pretrain_saver.save(session, os.path.join(params.model, 'model_ppl_lstm_pretrain_11th_epoch') + '/model_ppl_lstm_pretrain', global_step=1)
					with open(os.path.join(log_dir, "training_info_ppl_pretrain.txt"), "a") as f:
						f.write("Saving model at 11th epoch.\n")
				
				# Early stopping
				#if patience_count > params.pretrain_patience:
				if patience_count > params.patience:
					print("Early stopping criterion satisfied.")
					break
		
		# Restoring the saved pretraining variables values
		if ppl_model:
			pretrain_saver.restore(session, tf.train.latest_checkpoint(model_dir_ppl_lstm))

			print("LSTM PPL model restored.")

			## Validation set PPL
			total_val_nll, total_val_ppl = self.run_epoch(
				dataset.batches_split('validation_lstm_sents', params.batch_size, num_epochs=1, shuffle=False, multilabel=params.multi_label),
				params,
				session
			)

			print('LSTM val PPL: {:.3f},	LSTM val NLL: {:.3f}'.format(
				total_val_ppl,
				total_val_nll
			))

			## Test set PPL
			total_test_nll, total_test_ppl = self.run_epoch(
				dataset.batches_split('test_lstm_sents', params.batch_size, num_epochs=1, shuffle=False, multilabel=params.multi_label),
				params,
				session
			)

			print('LSTM val PPL: {:.3f},	LSTM val NLL: {:.3f}'.format(
				total_test_ppl,
				total_test_nll
			))

			# logging information
			with open(os.path.join(log_dir, "training_info_ppl_pretrain.txt"), "a") as f:
				f.write("LSTM val PPL: %s,	LSTM val NLL: %s\n" % (total_val_ppl, total_val_nll))
				f.write("LSTM test PPL: %s,	LSTM test NLL: %s\n" % (total_test_ppl, total_test_nll))

			print("Restored LSTM PPL result.")


	def run_epoch(self, data, params, session):
		this_nll = []
		this_loss_normed = []
		this_num_words = []
		
		for y_rnn, x_rnn, rnn_seq_lengths, _, _, _ in data:
			if params.use_bilm:
				x_rnn_input = x_rnn
				x_rnn_output = x_rnn[:, 1:-1]
			else:
				x_rnn_input = x_rnn[:, :-1]
				x_rnn_output = x_rnn[:, 1:]
				rnn_seq_lengths -= 1

			feed_dict = {
				self.x_rnn_input: x_rnn_input,
				self.x_rnn_output: x_rnn_output,
				self.y_rnn: y_rnn,
				self.rnn_seq_lengths: rnn_seq_lengths,
				self.lstm_dropout_keep_prob: 1.0
			}

			if params.use_char_embeddings:
				x_rnn_char_input, rnn_char_seq_lengths = get_char_indices(x_rnn_input, vocab_rnn_word, vocab_rnn_char)
				feed_dict[self.x_rnn_char_input] = x_rnn_char_input
				feed_dict[self.rnn_char_seq_lengths] = rnn_char_seq_lengths
			
			loss_normed, loss_unnormed, mask = session.run([self.loss_normed_lstm, self.loss_unnormed_lstm, self.mask_lstm], feed_dict=feed_dict)

			this_nll.extend(loss_unnormed)
			this_loss_normed.extend(loss_normed)
			this_num_words.append(np.sum(mask))
		#import pdb; pdb.set_trace()
		total_nll = np.mean(this_nll)
		total_words = np.sum(this_num_words)
		total_ppl = np.exp(np.sum(this_nll) / total_words)

		return total_nll, total_ppl


	def run_epoch_comb_docnade(self, data_LM, data_TM, TM_vocab, LM_vocab, params, session):
		this_nll = []
		this_loss_normed = []
		this_lstm_words = []
		
		for y, x, seq_lengths in data_TM:
			y_rnn, x_rnn, rnn_seq_lengths, split_indices, x_rnn_original, seq_lengths_original = next(data_LM)
			if params.use_bilm:
				x_rnn_input = x_rnn
				x_rnn_output = x_rnn[:, 1:-1]
			else:
				x_rnn_input = x_rnn[:, :-1]
				x_rnn_output = x_rnn[:, 1:]
				rnn_seq_lengths -= 1

			x, y, seq_lengths = data.split_data_TM(x, y, seq_lengths, split_indices)
			
			feed_dict = {
				self.topic_model.x: x,
				self.topic_model.y: y,
				self.topic_model.seq_lengths: seq_lengths,
			}

			if params.use_sent_topic_rep:
				pad_length = max_pad_length(TM_vocab, LM_vocab, x_rnn, rnn_seq_lengths)
				x_rnn_new = get_sent_topic_reps_docnade(TM_vocab, LM_vocab, x_rnn, rnn_seq_lengths, pad_length=pad_length, pad_value=len(TM_vocab))
				feed_dict[self.topic_model.x_sent] = x_rnn_new

			feed_dict[self.x_rnn_input] = x_rnn_input
			feed_dict[self.x_rnn_output] = x_rnn_output
			feed_dict[self.y_rnn] = y_rnn
			feed_dict[self.rnn_seq_lengths] = rnn_seq_lengths
			feed_dict[self.lstm_dropout_keep_prob] = 1.0

			if params.use_char_embeddings:
				x_rnn_char_input, rnn_char_seq_lengths = get_char_indices(x_rnn_input, vocab_rnn_word, vocab_rnn_char)
				feed_dict[self.x_rnn_char_input] = x_rnn_char_input
				feed_dict[self.rnn_char_seq_lengths] = rnn_char_seq_lengths
		
			if params.supervised:
				sys.exit()
			else:
				loss_normed, loss_unnormed, lstm_mask \
					= session.run([self.loss_normed_comb, 
									self.loss_unnormed_comb, 
									self.mask_comb], 
									feed_dict=feed_dict)
				this_nll.extend(loss_unnormed)
				this_loss_normed.extend(loss_normed)
				this_lstm_words.append(np.sum(lstm_mask))
		
		total_nll = np.mean(this_nll)
		total_lstm_ppl = np.exp(np.sum(this_nll) / np.sum(this_lstm_words))

		return total_nll, total_lstm_ppl


	def run_epoch_comb_nvdm(self, data_LM, data_TM, TM_vocab, LM_vocab, params, session):
		this_nll = []
		this_loss_normed = []
		this_lstm_words = []
		
		for counter, (y, x, count, mask) in enumerate(data_TM):
			#if counter == 40:
			#	import pdb; pdb.set_trace()
			y_rnn, x_rnn, rnn_seq_lengths, split_indices, x_rnn_original, seq_lengths_original = next(data_LM)
			if params.use_bilm:
				x_rnn_input = x_rnn
				x_rnn_output = x_rnn[:, 1:-1]
			else:
				x_rnn_input = x_rnn[:, :-1]
				x_rnn_output = x_rnn[:, 1:]
				rnn_seq_lengths -= 1

			if params.use_topic_embedding:
				x_doc_mask = get_doc_topic_mask_nvdm(TM_vocab, LM_vocab, x, x_rnn_original)
				#x_doc_mask = get_doc_topic_mask_nvdm(TM_vocab, LM_vocab, x, x_rnn)
				x_doc_mask = x_doc_mask[split_indices]
			
			x = x[split_indices]
			mask = mask[split_indices]
			y = [y[index] for index in split_indices]
			count = [count[index] for index in split_indices]
			
			feed_dict = {
				self.topic_model.x.name: x,
				self.topic_model.mask.name: mask,
				self.topic_model.input_batch_size: x.shape[0]
			}

			if params.use_topic_embedding:
				feed_dict[self.topic_model.x_doc_mask] = x_doc_mask

			if params.use_sent_topic_rep:
				x_rnn_new = get_sent_topic_reps_nvdm(TM_vocab, LM_vocab, x_rnn, rnn_seq_lengths)
				feed_dict[self.topic_model.x_sent] = x_rnn_new

			feed_dict[self.x_rnn_input] = x_rnn_input
			feed_dict[self.x_rnn_output] = x_rnn_output
			feed_dict[self.y_rnn] = y_rnn
			feed_dict[self.rnn_seq_lengths] = rnn_seq_lengths
			feed_dict[self.lstm_dropout_keep_prob] = 1.0
			feed_dict[self.tm_dropout_keep_prob] = 1.0

			if params.use_char_embeddings:
				x_rnn_char_input, rnn_char_seq_lengths = get_char_indices(x_rnn_input, vocab_rnn_word, vocab_rnn_char)
				feed_dict[self.x_rnn_char_input] = x_rnn_char_input
				feed_dict[self.rnn_char_seq_lengths] = rnn_char_seq_lengths
		
			if params.supervised:
				sys.exit()
			else:
				loss_normed, loss_unnormed, lstm_mask \
					= session.run([self.loss_normed_comb, 
									self.loss_unnormed_comb, 
									self.mask_comb], 
									feed_dict=feed_dict)
				this_nll.extend(loss_unnormed)
				this_loss_normed.extend(loss_normed)
				this_lstm_words.append(np.sum(lstm_mask))
		#np.save('/home/ubuntu/TM_LM_code/model/apnews_val_nll_list_d_s_p_e.npy', np.array(this_nll, dtype=np.float32))
		total_nll = np.mean(this_nll)
		total_lstm_ppl = np.exp(np.sum(this_nll) / np.sum(this_lstm_words))

		return total_nll, total_lstm_ppl

	def get_logits_comb_nvdm(self, data_LM, data_TM, TM_vocab, LM_vocab, params, session):
		pred_indices = []
		true_indices = []
		
		for counter, (y, x, count, mask) in enumerate(data_TM):
			if counter % 1000 == 0:
				print("batch: %s" % (counter))
			y_rnn, x_rnn, rnn_seq_lengths, split_indices, x_rnn_original, seq_lengths_original = next(data_LM)
			if params.use_bilm:
				x_rnn_input = x_rnn
				x_rnn_output = x_rnn[:, 1:-1]
			else:
				x_rnn_input = x_rnn[:, :-1]
				x_rnn_output = x_rnn[:, 1:]
				rnn_seq_lengths -= 1

			if params.use_topic_embedding:
				x_doc_mask = get_doc_topic_mask_nvdm(TM_vocab, LM_vocab, x, x_rnn)
			
			feed_dict = {
				self.topic_model.x.name: x,
				self.topic_model.mask.name: mask,
				self.topic_model.input_batch_size: x.shape[0]
			}

			if params.use_topic_embedding:
				feed_dict[self.topic_model.x_doc_mask] = x_doc_mask

			if params.use_sent_topic_rep:
				x_rnn_new = get_sent_topic_reps_nvdm(TM_vocab, LM_vocab, x_rnn, rnn_seq_lengths)
				feed_dict[self.topic_model.x_sent] = x_rnn_new

			feed_dict[self.x_rnn_input] = x_rnn_input
			feed_dict[self.x_rnn_output] = x_rnn_output
			feed_dict[self.y_rnn] = y_rnn
			feed_dict[self.rnn_seq_lengths] = rnn_seq_lengths
			feed_dict[self.lstm_dropout_keep_prob] = 1.0
			feed_dict[self.tm_dropout_keep_prob] = 1.0

			if params.use_char_embeddings:
				x_rnn_char_input, rnn_char_seq_lengths = get_char_indices(x_rnn_input, vocab_rnn_word, vocab_rnn_char)
				feed_dict[self.x_rnn_char_input] = x_rnn_char_input
				feed_dict[self.rnn_char_seq_lengths] = rnn_char_seq_lengths
		
			if params.supervised:
				sys.exit()
			else:
				logits = session.run(self.logits_comb, 
									feed_dict=feed_dict)
				#import pdb; pdb.set_trace()
				logits = np.reshape(logits, (x_rnn_input.shape[0], x_rnn_input.shape[1], -1))
				pred_indices.extend([list(index_list) for index_list in np.argmax(logits, axis=2)])
				true_indices.extend([list(index_list) for index_list in x_rnn_output])
		import pdb; pdb.set_trace()
		true_words = []
		for true_list in true_indices:
			true_words.append([LM_vocab[index] for index in true_list])

		pred_words = []
		for pred_list in pred_indices:
			pred_words.append([LM_vocab[index] for index in pred_list])
		
		return pred_words, true_words

	def get_logits(self, data, params, session, LM_vocab):
		pred_indices = []
		true_indices = []
		
		counter = 0
		for y_rnn, x_rnn, rnn_seq_lengths, _, _, _ in data:
			if counter % 1000 == 0:
				print("batch: %s" % (counter))
			if params.use_bilm:
				x_rnn_input = x_rnn
				x_rnn_output = x_rnn[:, 1:-1]
			else:
				x_rnn_input = x_rnn[:, :-1]
				x_rnn_output = x_rnn[:, 1:]
				rnn_seq_lengths -= 1

			feed_dict = {
				self.x_rnn_input: x_rnn_input,
				self.x_rnn_output: x_rnn_output,
				self.y_rnn: y_rnn,
				self.rnn_seq_lengths: rnn_seq_lengths,
				self.lstm_dropout_keep_prob: 1.0
			}

			if params.use_char_embeddings:
				x_rnn_char_input, rnn_char_seq_lengths = get_char_indices(x_rnn_input, vocab_rnn_word, vocab_rnn_char)
				feed_dict[self.x_rnn_char_input] = x_rnn_char_input
				feed_dict[self.rnn_char_seq_lengths] = rnn_char_seq_lengths
			
			logits = session.run(self.logits_pretrain, feed_dict=feed_dict)

			logits = np.reshape(logits, (x_rnn_input.shape[0], x_rnn_input.shape[1], -1))
			pred_indices.extend([list(index_list) for index_list in np.argmax(logits, axis=2)])
			true_indices.extend([list(index_list) for index_list in x_rnn_output])
			
			counter += 1
		
		true_words = []
		for true_list in true_indices:
			true_words.append([LM_vocab[index] for index in true_list])

		pred_words = []
		for pred_list in pred_indices:
			pred_words.append([LM_vocab[index] for index in pred_list])

		return pred_words, true_words
	
	def hidden_vectors_comb(self, data, data_lstm, params, session):
		if params.TM_type == "docnade":
			vecs = self.hidden_vectors_comb_docnade(data, data_lstm, params, session)
		if params.TM_type == "nvdm":
			vecs = self.hidden_vectors_comb_nvdm(data, data_lstm, params, session)
		return vecs


	def hidden_vectors_comb_docnade(self, data_TM, data_LM, TM_vocab, LM_vocab, params, session):
		vecs = []
		for y, x, seq_lengths in data_TM:
			feed_dict = {
				self.topic_model.x: x,
				self.topic_model.y: y,
				self.topic_model.seq_lengths: seq_lengths,
			}

			y_rnn, x_rnn, rnn_seq_lengths = next(data_LM)
			if params.use_bilm:
				x_rnn_input = x_rnn
				x_rnn_output = x_rnn[:, 1:-1]
			else:
				x_rnn_input = x_rnn[:, :-1]
				x_rnn_output = x_rnn[:, 1:]
				rnn_seq_lengths -= 1

			#if params.use_sent_topic_rep:
			#	pad_length = max_pad_length(TM_vocab, LM_vocab, x_rnn, rnn_seq_lengths)
			#	x_rnn_new = get_sent_topic_reps_docnade(TM_vocab, LM_vocab, x_rnn, rnn_seq_lengths, pad_length=pad_length, pad_value=len(TM_vocab))
			#	feed_dict[self.topic_model.x_sent] = x_rnn_new

			feed_dict[self.x_rnn_input] = x_rnn_input
			feed_dict[self.x_rnn_output] = x_rnn_output
			feed_dict[self.y_rnn] = y_rnn
			feed_dict[self.rnn_seq_lengths] = rnn_seq_lengths
			feed_dict[self.lstm_dropout_keep_prob] = 1.0

			if params.use_char_embeddings:
				x_rnn_char_input, rnn_char_seq_lengths = get_char_indices(x_rnn_input, vocab_rnn_word, vocab_rnn_char)
				feed_dict[self.x_rnn_char_input] = x_rnn_char_input
				feed_dict[self.rnn_char_seq_lengths] = rnn_char_seq_lengths8
			
			if (params.combination_type == 'concat') or (params.combination_type == 'sum') or (params.combination_type == 'gating'):
				vecs.extend(
					session.run([self.last_comb_h], feed_dict=feed_dict)[0]
				)
			elif params.combination_type == 'projected':
				vecs.extend(
					session.run([self.last_comb_h_proj], feed_dict=feed_dict)[0]
				)
			else:
				print("Wrong value for params.combination_type: ", params.combination_type)
				sys.exit()
			
		return np.array(vecs)


	def hidden_vectors_comb_nvdm(self, data, data_lstm, TM_vocab, LM_vocab, params, session):
		vecs = []
		for y, x, count, mask in data:
			y_rnn, x_rnn, rnn_seq_lengths = next(data_lstm)
			if params.use_bilm:
				x_rnn_input = x_rnn
				x_rnn_output = x_rnn[:, 1:-1]
			else:
				x_rnn_input = x_rnn[:, :-1]
				x_rnn_output = x_rnn[:, 1:]
				rnn_seq_lengths -= 1

			feed_dict = {
				self.topic_model.x.name: x,
				self.topic_model.mask.name: mask,
				self.topic_model.input_batch_size: x.shape[0]
			}

			#if params.use_sent_topic_rep:
			#	x_rnn_new = get_sent_topic_reps_nvdm(TM_vocab, LM_vocab, x_rnn, rnn_seq_lengths)
			#	feed_dict[self.topic_model.x_sent] = x_rnn_new

			feed_dict[self.x_rnn_input] = x_rnn_input
			feed_dict[self.x_rnn_output] = x_rnn_output
			feed_dict[self.y_rnn] = y_rnn
			feed_dict[self.rnn_seq_lengths] = rnn_seq_lengths
			feed_dict[self.lstm_dropout_keep_prob] = 1.0

			if params.use_char_embeddings:
				x_rnn_char_input, rnn_char_seq_lengths = get_char_indices(x_rnn_input, vocab_rnn_word, vocab_rnn_char)
				feed_dict[self.x_rnn_char_input] = x_rnn_char_input
				feed_dict[self.rnn_char_seq_lengths] = rnn_char_seq_lengths8
			
			if (params.combination_type == 'concat') or (params.combination_type == 'sum') or (params.combination_type == 'gating'):
				vecs.extend(
					session.run([self.last_comb_h], feed_dict=feed_dict)[0]
				)
			elif params.combination_type == 'projected':
				vecs.extend(
					session.run([self.last_comb_h_proj], feed_dict=feed_dict)[0]
				)
			else:
				print("Wrong value for params.combination_type: ", params.combination_type)
				sys.exit()
			
		return np.array(vecs)


	def hidden_vectors_lstm(self, data_lstm, params, session):
		vecs = []
		
		#for y_rnn, x_rnn, rnn_seq_lengths in data_lstm:
		for y_rnn, x_rnn, rnn_seq_lengths, _, _, _ in data_lstm:
			feed_dict = {}
			
			feed_dict[self.x_rnn_input] = x_rnn
			#feed_dict[self.y_rnn] = y_rnn
			feed_dict[self.rnn_seq_lengths] = rnn_seq_lengths
			feed_dict[self.lstm_dropout_keep_prob] = 1.0

			if params.use_char_embeddings:
				x_rnn_char_input, rnn_char_seq_lengths = get_char_indices(x_rnn_input, vocab_rnn_word, vocab_rnn_char)
				feed_dict[self.x_rnn_char_input] = x_rnn_char_input
				feed_dict[self.rnn_char_seq_lengths] = rnn_char_seq_lengths

			vecs.extend(
				session.run([self.last_lstm_h], feed_dict=feed_dict)[0]
			)
		
		return np.array(vecs)

	def hidden_vectors_all(self, data_lm, data_tm, TM_vocab, LM_vocab, params, session):
		#vecs_lm = []
		#vecs_tm = []
		vecs_concat = []
		vecs_proj = []

		for y, x, count, mask in data_tm:
			y_rnn, x_rnn, rnn_seq_lengths, _, _, _ = next(data_lm)
			x_rnn_input = x_rnn

			feed_dict = {
				self.topic_model.x.name: x,
				self.topic_model.mask.name: mask,
				self.topic_model.input_batch_size: x.shape[0]
			}

			feed_dict[self.x_rnn_input] = x_rnn_input
			#feed_dict[self.x_rnn_output] = x_rnn_output
			#feed_dict[self.y_rnn] = y_rnn
			feed_dict[self.rnn_seq_lengths] = rnn_seq_lengths
			feed_dict[self.lstm_dropout_keep_prob] = 1.0

			if params.use_topic_embedding:
				x_doc_mask = get_doc_topic_mask_nvdm(TM_vocab, LM_vocab, x, x_rnn)
				feed_dict[self.topic_model.x_doc_mask] = x_doc_mask

			if params.use_sent_topic_rep:
				x_rnn_new = get_sent_topic_reps_nvdm(TM_vocab, LM_vocab, x_rnn, rnn_seq_lengths)
				feed_dict[self.topic_model.x_sent] = x_rnn_new
			#import pdb; pdb.set_trace()
			"""
			vec_lm, vec_tm, vec_concat, vec_proj \
				= session.run([self.last_lstm_h,
								self.topic_model.last_h,
								self.last_comb_h,
								self.last_comb_h_proj], 
								feed_dict=feed_dict)
			"""
			vec_concat, vec_proj \
				= session.run([self.last_comb_h, self.last_comb_h_proj], 
							   feed_dict=feed_dict)
			#vecs_lm.append(vec_lm)
			#vecs_tm.append(vec_tm)
			vecs_concat.append(vec_concat)
			vecs_proj.append(vec_proj)
		
		#final_lm_vecs = np.concatenate(vecs_lm, axis=0)
		#final_tm_vecs = np.concatenate(vecs_tm, axis=0)
		final_concat_vecs = np.concatenate(vecs_concat, axis=0)
		final_proj_vecs = np.concatenate(vecs_proj, axis=0)

		#return final_lm_vecs, final_tm_vecs, final_concat_vecs, final_proj_vecs
		return final_concat_vecs, final_proj_vecs

def run_epoch_lstm_reload(data, params, graph, session):
	this_nll = []
	this_loss_normed = []
	this_num_words = []

	x_rnn_input_ph = graph.get_tensor_by_name("x_rnn_input:0")
	x_rnn_output_ph = graph.get_tensor_by_name("x_rnn_output:0")
	y_rnn_ph = graph.get_tensor_by_name("y_rnn:0")
	rnn_seq_lengths_ph = graph.get_tensor_by_name("rnn_seq_lengths:0")
	lstm_dropout_keep_prob_ph = graph.get_tensor_by_name("lstm_dropout_keep_prob:0")

	if params['use_char_embeddings']:
		x_rnn_char_input_ph = graph.get_tensor_by_name("x_char_rnn:0")
		rnn_char_seq_lengths_ph = graph.get_tensor_by_name("rnn_char_seq_lengths:0")

	loss_unnormed_lstm = graph.get_tensor_by_name("LM_loss_unnormed:0")
	loss_normed_lstm = graph.get_tensor_by_name("LM_loss_normed:0")
	mask_lstm = graph.get_tensor_by_name("LM_loss_mask:0")
	
	for y_rnn, x_rnn, rnn_seq_lengths, _, _, _ in data:
		#import pdb; pdb.set_trace()
		if params['use_bilm']:
			x_rnn_input = x_rnn
			x_rnn_output = x_rnn[:, 1:-1]
		else:
			x_rnn_input = x_rnn[:, :-1]
			x_rnn_output = x_rnn[:, 1:]
			rnn_seq_lengths -= 1

		feed_dict = {
			x_rnn_input_ph: x_rnn_input,
			x_rnn_output_ph: x_rnn_output,
			y_rnn_ph: y_rnn,
			rnn_seq_lengths_ph: rnn_seq_lengths,
			lstm_dropout_keep_prob_ph: 1.0
		}

		if params['use_char_embeddings']:
			x_rnn_char_input, rnn_char_seq_lengths = get_char_indices(x_rnn_input, vocab_rnn_word, vocab_rnn_char)
			feed_dict[x_rnn_char_input_ph] = x_rnn_char_input
			feed_dict[rnn_char_seq_lengths_ph] = rnn_char_seq_lengths
		
		loss_normed, loss_unnormed, mask = session.run([loss_normed_lstm, loss_unnormed_lstm, mask_lstm], feed_dict=feed_dict)

		this_nll.extend(loss_unnormed)
		this_loss_normed.extend(loss_normed)
		this_num_words.append(np.sum(mask))

	total_nll = np.mean(this_nll)
	total_words = np.sum(this_num_words)
	total_ppl = np.exp(np.sum(this_nll) / total_words)

	return total_nll, total_ppl

def run_epoch_comb_nvdm_reload(data_LM, data_TM, TM_vocab, LM_vocab, params, graph, session):
	this_nll = []
	this_loss_normed = []
	this_lstm_words = []

	x_rnn_input_ph = graph.get_tensor_by_name("x_rnn_input:0")
	x_rnn_output_ph = graph.get_tensor_by_name("x_rnn_output:0")
	y_rnn_ph = graph.get_tensor_by_name("y_rnn:0")
	rnn_seq_lengths_ph = graph.get_tensor_by_name("rnn_seq_lengths:0")
	lstm_dropout_keep_prob_ph = graph.get_tensor_by_name("lstm_dropout_keep_prob:0")

	if params['use_char_embeddings']:
		x_rnn_char_input_ph = graph.get_tensor_by_name("x_char_rnn:0")
		rnn_char_seq_lengths_ph = graph.get_tensor_by_name("rnn_char_seq_lengths:0")

	loss_unnormed_lstm = graph.get_tensor_by_name("TM_LM_loss_unnormed:0")
	loss_normed_lstm = graph.get_tensor_by_name("TM_LM_loss_normed:0")
	mask_lstm = graph.get_tensor_by_name("TM_LM_loss_mask:0")

	x_nvdm_ph = graph.get_tensor_by_name("x:0")
	mask_nvdm_ph = graph.get_tensor_by_name("mask:0")

	if params['use_sent_topic_rep']:
		x_sent_nvdm_ph = graph.get_tensor_by_name("x_sent:0")

	if params['use_topic_embedding']:
		x_doc_mask_nvdm_ph = graph.get_tensor_by_name("x_doc_mask:0")
	#import pdb; pdb.set_trace()
	for counter, (y, x, count, mask) in enumerate(data_TM):
		#if counter == 40:
		#	import pdb; pdb.set_trace()
		y_rnn, x_rnn, rnn_seq_lengths, split_indices, x_rnn_original, seq_lengths_original = next(data_LM)
		if params['use_bilm']:
			x_rnn_input = x_rnn
			x_rnn_output = x_rnn[:, 1:-1]
		else:
			x_rnn_input = x_rnn[:, :-1]
			x_rnn_output = x_rnn[:, 1:]
			rnn_seq_lengths -= 1

		if params['use_topic_embedding']:
			x_doc_mask = get_doc_topic_mask_nvdm(TM_vocab, LM_vocab, x, x_rnn_original)
			x_doc_mask = x_doc_mask[split_indices]
		
		x = x[split_indices]
		mask = mask[split_indices]
		y = [y[index] for index in split_indices]
		count = [count[index] for index in split_indices]
		
		feed_dict = {
			x_nvdm_ph: x,
			mask_nvdm_ph: mask,
		}

		if params['use_topic_embedding']:
			feed_dict[x_doc_mask_nvdm_ph] = x_doc_mask

		if params['use_sent_topic_rep']:
			x_rnn_new = get_sent_topic_reps_nvdm(TM_vocab, LM_vocab, x_rnn, rnn_seq_lengths)
			feed_dict[x_sent_nvdm_ph] = x_rnn_new

		feed_dict[x_rnn_input_ph] = x_rnn_input
		feed_dict[x_rnn_output_ph] = x_rnn_output
		feed_dict[y_rnn_ph] = y_rnn
		feed_dict[rnn_seq_lengths_ph] = rnn_seq_lengths
		feed_dict[lstm_dropout_keep_prob_ph] = 1.0

		if params['use_char_embeddings']:
			x_rnn_char_input, rnn_char_seq_lengths = get_char_indices(x_rnn_input, vocab_rnn_word, vocab_rnn_char)
			feed_dict[x_rnn_char_input_ph] = x_rnn_char_input
			feed_dict[rnn_char_seq_lengths_ph] = rnn_char_seq_lengths
	
		if params['supervised']:
			sys.exit()
		else:
			loss_normed, loss_unnormed, lstm_mask \
				= session.run([loss_normed_lstm, 
								loss_unnormed_lstm, 
								mask_lstm], 
								feed_dict=feed_dict)
			this_nll.extend(loss_unnormed)
			this_loss_normed.extend(loss_normed)
			this_lstm_words.append(np.sum(lstm_mask))
	
	total_nll = np.mean(this_nll)
	total_lstm_ppl = np.exp(np.sum(this_nll) / np.sum(this_lstm_words))

	return total_nll, total_lstm_ppl