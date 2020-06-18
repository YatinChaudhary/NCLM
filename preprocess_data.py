
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, random
import os

# sys.setdefaultencoding() does not exist, here!
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')

from nltk import word_tokenize
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re, random, pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models, similarities
from os.path import expanduser
from collections import defaultdict
import codecs

import csv
from nltk.tokenize import RegexpTokenizer
import numpy as np
import tensorflow as tf
import argparse

import model.data_lstm
import operator
from collections import Counter
import copy

#from stanfordcorenlp import StanfordCoreNLP
#tokenizer_path = "/home/ubuntu/stanford-corenlp-full-2018-02-27"
#tokenizer_path = "../../stanford-corenlp-full-2018-02-27"

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

csv.field_size_limit(2**28)
home_dir = os.getenv("HOME")

with open("en.txt", "r") as f:
	cachedStopWords = [line.strip() for line in f.readlines()]

##########################################################################################

def update_vocab(symbol, idxvocab, vocabxid):
	idxvocab.append(symbol)
	vocabxid[symbol] = len(idxvocab) - 1 

def gen_vocab_docnade(dummy_symbols, corpus, stopwords, vocab_minfreq, vocab_maxfreq, verbose, tokenizer):
	idxvocab = []
	vocabxid = defaultdict(int)
	vocab_freq = defaultdict(int)
	
	total_doc_tokens = []
	docnade_doc_tokens = []
	for doc_id, doc in enumerate(corpus):
		temp_doc_tokens = []
		doc_tokens = []
		for doc_sent in doc.strip().lower().split("\t"):
			sent_tokens = doc_sent.strip().lower().split()
			doc_tokens.extend(sent_tokens)
			temp_doc_tokens.append(sent_tokens)
		total_doc_tokens.append(temp_doc_tokens)
		
		temp_docnade_tokens = []
		for i, tokens in enumerate(temp_doc_tokens):
			temp_tokens = copy.deepcopy(temp_doc_tokens)
			temp_tokens.pop(i)
			temp_docnade_tokens.append(temp_tokens)
		docnade_doc_tokens.append(temp_docnade_tokens)

		for word in doc_tokens:
			vocab_freq[word] += 1
		if doc_id % 1000 == 0 and verbose:
			sys.stdout.write(str(doc_id) + " processed\r")
			sys.stdout.flush()

	#add in dummy symbols into vocab
	for s in dummy_symbols:
		update_vocab(s, idxvocab, vocabxid)

	TM_ignore = []
	#remove low fequency words
	for w, f in sorted(vocab_freq.items(), key=operator.itemgetter(1), reverse=True):
		if f < vocab_minfreq:
			break
		else:
			if f < 100:
				TM_ignore.append(w)
			update_vocab(w, idxvocab, vocabxid)

	#ignore stopwords, frequent words and symbols for the document input for topic model
	stopwords = set([item.strip().lower() for item in stopwords])
	freqwords = set([item[0] for item in sorted(vocab_freq.items(), key=operator.itemgetter(1), \
		reverse=True)[:int(float(len(vocab_freq))*vocab_maxfreq)]]) #ignore top N% most frequent words for topic model
	alpha_check = re.compile("[a-zA-Z]")
	symbols = set([ w for w in vocabxid.keys() if ((alpha_check.search(w) == None) or w.startswith("'")) ])
	ignore = stopwords | freqwords | symbols | set(dummy_symbols) | set(["n't"])
	ignore = [vocabxid[w] for w in ignore if w in vocabxid]
	ignore.extend([vocabxid[w] for w in TM_ignore])
	ignore = set(ignore)
	return idxvocab, vocabxid, ignore, total_doc_tokens, docnade_doc_tokens


def get_tokens(corpus):
	total_doc_tokens = []
	docnade_doc_tokens = []
	for doc_id, doc in enumerate(corpus):
		temp_doc_tokens = []
		for doc_sent in doc.strip().lower().split("\t"):
			sent_tokens = doc_sent.strip().lower().split()
			temp_doc_tokens.append(sent_tokens)
		total_doc_tokens.append(temp_doc_tokens)
		
		temp_docnade_tokens = []
		for i, tokens in enumerate(temp_doc_tokens):
			temp_tokens = copy.deepcopy(temp_doc_tokens)
			temp_tokens.pop(i)
			temp_docnade_tokens.append(temp_tokens)
		docnade_doc_tokens.append(temp_docnade_tokens)

	return total_doc_tokens, docnade_doc_tokens

##########################################################################################

def loadGloveModel(gloveFile=None, hidden_size=None):
	if gloveFile is None:
		if hidden_size == 50:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.50d.txt")
		elif hidden_size == 100:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.100d.txt")
		elif hidden_size == 200:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.200d.txt")
		elif hidden_size == 300:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.300d.txt")
		else:
			print('Invalid dimension [%d] for Glove pretrained embedding matrix!!' % (hidden_size))
			exit()

	print("Loading Glove Model")
	f = open(gloveFile, 'r')
	model = {}
	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		embedding = np.array([float(val) for val in splitLine[1:]])
		model[word] = embedding
	print("Done.", len(model), " words loaded!")
	return model


def tokens(text, tokenizer):
	return [w.lower().strip() for w in text.split()]


def counts_to_sequence(counts):
	seq = []
	for i in range(len(counts)):
		seq.extend([i] * int(counts[i]))
	return seq


def log_counts(ids, vocab_size):
	counts = np.bincount(ids, minlength=vocab_size)
	return np.floor(0.5 + np.log(counts + 1))


def preprocess_docnade(text, vocab_to_id, tokenizer):
	ids = [vocab_to_id.get(x) for x in tokens(text, tokenizer) if not (vocab_to_id.get(x) is None)]
	
	counts = log_counts(ids, len(vocab_to_id))
	sequence = counts_to_sequence(counts)
	
	if len(sequence) == 0:
		return None
	else:
		return ' '.join([str(x) for x in sequence])

def preprocess_nvdm(text, vocab_to_id, tokenizer):
	ids = [vocab_to_id.get(x) for x in tokens(text, tokenizer) if not (vocab_to_id.get(x) is None)]
	
	if len(ids) == 0:
		return None
	else:
		return ' '.join([str(x) for x in ids])

def preprocess_lstm(text, vocab_to_id, tokenizer):
	ids = []
	for x in tokens(text, tokenizer):
		if vocab_to_id.get(x.strip()) is None:
			ids.append(vocab_to_id.get("_unk_"))
		else:
			ids.append(vocab_to_id.get(x.strip()))

	sequence = ids
	
	if len(sequence) == 0:
		return None
	else:
		return ' '.join([str(x) for x in sequence])

def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets.
	"""
	#string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	string = re.sub(r"\s+", " ", string)
	#return string.strip().split()
	return string


def TF(docs, tokenizer, max_features=2000):
	cv = CountVectorizer(tokenizer=tokenizer.word_tokenize, min_df=3, max_df=1.0, max_features=max_features, encoding='utf-8', decode_error='ignore')
	cv.fit(docs)
	return cv


def load_file(filename):
	"""
	Read the tab delimited file containing the labels and the docs.

	"""
	labels = []
	docs = []

	with open(filename) as f:
		for line in f:
			content = line.split('\t')

			if len(content) > 2:
				print('incorrect read')
				exit()

			if len(content[1]) == 0: continue

			docs.append(str(content[1]).strip('\r').strip('\n').strip('\r\n'))
			labels.append(content[0])

	return docs, labels


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
	use_labels = False
	use_stanford_tokenizer = False
	docnade_RV = False
	docnade_vocab_size = 5000
	
	if use_stanford_tokenizer:
		print("Error-1")
	else:
		tokenizer = None

	doc_train_filename = args.training_file
	doc_val_filename = args.validation_file
	doc_test_filename = args.test_file

	if use_labels:
		doc_train_labels_filename = args.training_labels_file
		doc_val_labels_filename = args.validation_labels_file
		doc_test_labels_filename = args.test_labels_file
	
	train_csv_filename = os.path.join(args.data_output, "training.csv")
	val_csv_filename = os.path.join(args.data_output, "validation.csv")
	test_csv_filename = os.path.join(args.data_output, "test.csv")

	train_original_docs_non_replicated_filename = os.path.join(args.data_output, "training_original_docs_non_replicated.txt")
	val_original_docs_non_replicated_filename = os.path.join(args.data_output, "validation_original_docs_non_replicated.txt")
	test_original_docs_non_replicated_filename = os.path.join(args.data_output, "test_original_docs_non_replicated.txt")

	train_original_docs_replicated_filename = os.path.join(args.data_output, "training_original_docs_replicated.txt")
	val_original_docs_replicated_filename = os.path.join(args.data_output, "validation_original_docs_replicated.txt")
	test_original_docs_replicated_filename = os.path.join(args.data_output, "test_original_docs_replicated.txt")

	train_original_docs_minus_sents_filename = os.path.join(args.data_output, "training_original_docs_minus_sents.txt")
	val_original_docs_minus_sents_filename = os.path.join(args.data_output, "validation_original_docs_minus_sents.txt")
	test_original_docs_minus_sents_filename = os.path.join(args.data_output, "test_original_docs_minus_sents.txt")

	train_original_sents_filename = os.path.join(args.data_output, "training_original_sents.txt")
	val_original_sents_filename = os.path.join(args.data_output, "validation_original_sents.txt")
	test_original_sents_filename = os.path.join(args.data_output, "test_original_sents.txt")

	if not os.path.exists(args.data_output):
		os.makedirs(args.data_output)

	docnade_vocabulary = args.vocab_size
	docnade_vocab_filename = os.path.join(args.data_output, "vocab_nvdm.vocab")
	lstm_vocab_filename = os.path.join(args.data_output, "vocab_lstm.vocab")
	mapping_dict_filename = os.path.join(args.data_output, "mapping_dict.pkl")

	
	with codecs.open(doc_train_filename, "r", "utf-8") as f:
		train_docs = [line.lower().strip() for line in f.readlines()]

	with codecs.open(doc_val_filename, "r", "utf-8") as f:
		val_docs = [line.lower().strip() for line in f.readlines()]

	with codecs.open(doc_test_filename, "r", "utf-8") as f:
		test_docs = [line.lower().strip() for line in f.readlines()]

	if use_labels:
		with codecs.open(doc_train_labels_filename, "r", "utf-8") as f:
			train_docs_labels = [line.lower().strip() for line in f.readlines()]

		with codecs.open(doc_val_labels_filename, "r", "utf-8") as f:
			val_docs_labels = [line.lower().strip() for line in f.readlines()]

		with codecs.open(doc_test_labels_filename, "r", "utf-8") as f:
			test_docs_labels = [line.lower().strip() for line in f.readlines()]
	else:
		train_docs_labels = len(train_docs) * [0]
		val_docs_labels = len(val_docs) * [1]
		test_docs_labels = len(test_docs) * [2]

	###########################################################################
	# Prepare CSV file

	total_docs = []
	total_docs.extend(train_docs)

	# Saving docnade and lstm vocabularies
	vocab_list_lstm, vocab_dict_lstm, ignore_lstm, total_doc_tokens, docnade_doc_tokens = gen_vocab_docnade(["_bos_", "_eos_", "_unk_"], total_docs, cachedStopWords, 10, 0.001, True, tokenizer)

	vocab_to_id_lstm = dict(zip(vocab_list_lstm, range(len(vocab_list_lstm))))
	vocab_list_docnade = [vocab_list_lstm[index] for index in range(len(vocab_list_lstm)) if not index in ignore_lstm]
	
	if docnade_RV:
		vocab_list_docnade = vocab_list_docnade[:docnade_vocab_size]
	else:
		vocab_list_docnade = vocab_list_docnade
	vocab_to_id_docnade = dict(zip(vocab_list_docnade, range(len(vocab_list_docnade))))

	with open(lstm_vocab_filename, "w") as f:
		f.write('\n'.join(vocab_list_lstm))

	with open(docnade_vocab_filename, "w") as f:
		f.write('\n'.join(vocab_list_docnade))
	
	with open(train_csv_filename, 'w') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',')
		for doc, label in zip(train_docs, train_docs_labels):
			li = [str(label).lower().strip(), str(doc).lower().strip()]
			filewriter.writerow(li)

	with open(val_csv_filename, 'w') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',')
		for doc, label in zip(val_docs, val_docs_labels):
			li = [str(label).lower().strip(), str(doc).lower().strip()]
			filewriter.writerow(li)
	
	with open(test_csv_filename, 'w') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',')
		for doc, label in zip(test_docs, test_docs_labels):
			li = [str(label).lower().strip(), str(doc).lower().strip()]
			filewriter.writerow(li)
	
	
	## Preparing full document CSV files for DocNADE Tensorflow
	data = model.data_lstm.Dataset(args.data_output)

	if not os.path.isdir(args.data_output):
		os.mkdir(args.data_output)

	labels = {}
	docnade_full_text_docs = {"training":[], "test":[], "validation":[]}
	removed_indices_docnade_full = {"training":[], "test":[], "validation":[]}

	for collection in data.collections:
		output_path = os.path.join(args.data_output, '{}_docnade_docs_non_replicated.csv'.format(collection))
		with open(output_path, 'w') as f:
			w = csv.writer(f, delimiter=',')
			count = -1
			for y, x in data.rows(collection, num_epochs=1):
				count += 1
				try:
					pre = preprocess_docnade(x, vocab_to_id_docnade, tokenizer)
					if pre is None:
						removed_indices_docnade_full[str(collection).lower()].append(count)
						continue
					if ':' in y:
						temp_labels = y.split(':')
						new_label = []
						for label in temp_labels:
							if label not in labels:
								labels[label] = len(labels)
							new_label.append(str(labels[label]))
						temp_label = ':'.join(new_label)
						w.writerow((temp_label, pre))
					else:
						if y not in labels:
							labels[y] = len(labels)
						w.writerow((labels[y], pre))
					
					docnade_full_text_docs[str(collection).lower()].append(str(y) + "\t" + x)
				except:
					import pdb; pdb.set_trace()

	# Saving training_original.txt, validation_original.txt, test_original.txt files

	with open(train_original_docs_non_replicated_filename, "w") as f:
		f.writelines('\n'.join(docnade_full_text_docs["training"]))

	with open(val_original_docs_non_replicated_filename, "w") as f:
		f.writelines('\n'.join(docnade_full_text_docs["validation"]))

	with open(test_original_docs_non_replicated_filename, "w") as f:
		f.writelines('\n'.join(docnade_full_text_docs["test"]))

	## Preparing CSV files for DocNADE-LSTM Tensorflow

	docnade_lstm_text_docs = {"training":[], "test":[], "validation":[]}
	removed_indices_docnade_lstm = {"training":[], "test":[], "validation":[]}

	for collection in data.collections:
		output_path = os.path.join(args.data_output, '{}_docnade_docs_minus_sents.csv'.format(collection))
		with open(output_path, 'w') as f:
			w = csv.writer(f, delimiter=',')
			count = -1
			for y, x in data.rows(collection, num_epochs=1):
				x_sents = x.split("\t")
				for i, x_sent in enumerate(x_sents):
					count += 1
					if len(x_sents) == 1:
						x_temp = x_sent
					else:
						x_sents_temp = copy.deepcopy(x_sents)
						x_sents_temp.pop(i)
						x_temp = "\t".join(x_sents_temp)
					try:
						pre = preprocess_docnade(x_temp, vocab_to_id_docnade, tokenizer)
						if pre is None:
							removed_indices_docnade_lstm[str(collection).lower()].append(count)
							continue
						if ':' in y:
							temp_labels = y.split(':')
							new_label = []
							for label in temp_labels:
								if label not in labels:
									labels[label] = len(labels)
								new_label.append(str(labels[label]))
							temp_label = ':'.join(new_label)
							w.writerow((temp_label, pre))
						else:
							if y not in labels:
								labels[y] = len(labels)
							w.writerow((labels[y], pre))
						
						docnade_lstm_text_docs[str(collection).lower()].append(str(y) + "\t" + x_temp)
					except:
						import pdb; pdb.set_trace()

	
	## Preparing full document CSV files for NVDM Tensorflow

	for collection in data.collections:
		output_path = os.path.join(args.data_output, '{}_nvdm_docs_non_replicated.csv'.format(collection))
		with open(output_path, 'w') as f:
			w = csv.writer(f, delimiter=',')
			count = -1
			for y, x in data.rows(collection, num_epochs=1):
				count += 1
				try:
					pre = preprocess_nvdm(x, vocab_to_id_docnade, tokenizer)
					if pre is None:
						continue
					if ':' in y:
						temp_labels = y.split(':')
						new_label = []
						for label in temp_labels:
							if label not in labels:
								labels[label] = len(labels)
							new_label.append(str(labels[label]))
						temp_label = ':'.join(new_label)
						w.writerow((temp_label, pre))
					else:
						if y not in labels:
							labels[y] = len(labels)
						w.writerow((labels[y], pre))
				except:
					import pdb; pdb.set_trace()

	## Preparing CSV files for NVDM-LSTM Tensorflow

	for collection in data.collections:
		output_path = os.path.join(args.data_output, '{}_nvdm_docs_minus_sents.csv'.format(collection))
		with open(output_path, 'w') as f:
			w = csv.writer(f, delimiter=',')
			count = -1
			for y, x in data.rows(collection, num_epochs=1):
				x_sents = x.split("\t")
				for i, x_sent in enumerate(x_sents):
					count += 1
					if len(x_sents) == 1:
						x_temp = x_sent
					else:
						x_sents_temp = copy.deepcopy(x_sents)
						x_sents_temp.pop(i)
						x_temp = "\t".join(x_sents_temp)
					try:
						pre = preprocess_nvdm(x_temp, vocab_to_id_docnade, tokenizer)
						if pre is None:
							continue
						if ':' in y:
							temp_labels = y.split(':')
							new_label = []
							for label in temp_labels:
								if label not in labels:
									labels[label] = len(labels)
								new_label.append(str(labels[label]))
							temp_label = ':'.join(new_label)
							w.writerow((temp_label, pre))
						else:
							if y not in labels:
								labels[y] = len(labels)
							w.writerow((labels[y], pre))
					except:
						import pdb; pdb.set_trace()

	## Preparing full document CSV files for LSTM Tensorflow

	for collection in data.collections:
		removed_indices_collection = removed_indices_docnade_full[str(collection).lower()]
		output_path = os.path.join(args.data_output, '{}_lstm_docs.csv'.format(collection))
		with open(output_path, 'w') as f:
			w = csv.writer(f, delimiter=',')
			count = -1
			for y, x in data.rows(collection, num_epochs=1):
				x = "_bos_ " + x + " _eos_"
				count += 1
				try:
					if count in removed_indices_collection:
						continue
					else:
						pre = preprocess_lstm(x, vocab_to_id_lstm, tokenizer)
					
					if ':' in y:
						temp_labels = y.split(':')
						new_label = []
						for label in temp_labels:
							if label not in labels:
								labels[label] = len(labels)
							new_label.append(str(labels[label]))
						temp_label = ':'.join(new_label)
						w.writerow((temp_label, pre))
					else:
						if y not in labels:
							labels[y] = len(labels)
						w.writerow((labels[y], pre))
				except:
					import pdb; pdb.set_trace()

	## Preparing CSV files for LSTM Tensorflow
	
	lstm_text_docs = {"training":[], "test":[], "validation":[]}

	for collection in data.collections:
		removed_indices_collection = removed_indices_docnade_lstm[str(collection).lower()]
		output_path = os.path.join(args.data_output, '{}_lstm_sents.csv'.format(collection))
		with open(output_path, 'w') as f:
			w = csv.writer(f, delimiter=',')
			count = -1
			for y, x in data.rows(collection, num_epochs=1):
				for x_sent in x.split("\t"):
					x_sent = "_bos_ " + x_sent + " _eos_"
					count += 1
					try:
						if count in removed_indices_collection:
							continue
						else:
							pre = preprocess_lstm(x_sent, vocab_to_id_lstm, tokenizer)
						
						if ':' in y:
							temp_labels = y.split(':')
							new_label = []
							for label in temp_labels:
								if label not in labels:
									labels[label] = len(labels)
								new_label.append(str(labels[label]))
							temp_label = ':'.join(new_label)
							w.writerow((temp_label, pre))
						else:
							if y not in labels:
								labels[y] = len(labels)
							w.writerow((labels[y], pre))
						
						lstm_text_docs[str(collection).lower()].append(str(y) + "\t" + x_sent)
					except:
						import pdb; pdb.set_trace()
		
	# Saving training_original.txt, validation_original.txt, test_original.txt files
	
	with open(train_original_sents_filename, "w") as f:
		f.writelines('\n'.join(lstm_text_docs["training"]))

	with open(val_original_sents_filename, "w") as f:
		f.writelines('\n'.join(lstm_text_docs["validation"]))
	
	with open(test_original_sents_filename, "w") as f:
		f.writelines('\n'.join(lstm_text_docs["test"]))
	
	# Writing labels file

	with open(os.path.join(args.data_output, 'labels.txt'), 'w') as f:
		f.write('\n'.join([k for k in sorted(labels, key=labels.get)]))
	

	if use_stanford_tokenizer:
		#tokenizer.close()
		print("Error-2")

	print("Done.")


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--training-file', type=str, required=True,
						help='path to training text file')
	parser.add_argument('--training-labels-file', type=str, default="",
						help='path to training labels text file')
	parser.add_argument('--validation-file', type=str, required=True,
						help='path to validation text file')
	parser.add_argument('--validation-labels-file', type=str, default="",
						help='path to validation labels text file')
	parser.add_argument('--test-file', type=str, required=True,
						help='path to test text file')
	parser.add_argument('--test-labels-file', type=str, default="",
						help='path to test labels text file')
	parser.add_argument('--data-output', type=str, required=True,
						help='path to data output directory')
	parser.add_argument('--vocab-size', type=int, default=2000,
						help='the vocab size')
	parser.add_argument('--split-train-val', type=str, default="False",
						help='whether to do train-val split')
	parser.add_argument('--split-num', type=int, default=50,
						help='number of documents in validation set')

	return parser.parse_args()


if __name__ == '__main__':
	main(parse_args())