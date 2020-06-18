"""
Author:		 Jey Han Lau
Date:		   May 2013
"""

import argparse
import sys
import os
import threading
import time
import codecs
from multiprocessing import Pool

from ComputeObservedCoherenceFunction import ComputeCoherence

#parser arguments
desc = "Computes the word pair co-occurrences for topics. Parallel processing is achieved by \
	splitting the corpus into multiple partitions."
parser = argparse.ArgumentParser(description=desc)
#####################
#positional argument#
#####################
parser.add_argument("topic_file", help="file that contains the topics")
parser.add_argument("ref_corpus_dir", help="directory that contains the reference corpus")
parser.add_argument("wordcount_file", help="file that contains corpus word counts as per topics")
parser.add_argument("coherence_file", help="file that contains coherence output")
"""
parser.add_argument("-t", "--topns", nargs="+", type=int, default=[10], \
    help="list of top-N topic words to consider for computing coherence; e.g. '-t 5 10' means it " + \
    " will compute coherence over top-5 words and top-10 words and then take the mean of both values." + \
    " Default = [10]")
"""
args = parser.parse_args()

#parameters
#window_size = 20 #size of the sliding window; 0 = use document as window
window_size = 0 #size of the sliding window; 0 = use document as window
colloc_sep = "_" #symbol for concatenating collocations
debug = False

#new parameters
topic_lemmatization = False
corpus_lemmatization = False

if topic_lemmatization or corpus_lemmatization:
	import spacy
	nlp = spacy.load("en_core_web_sm")

#constants
WTOTALKEY = "!!<TOTAL_WINDOWS>!!" #key name for total number of windows (in wordcount)

#input
#topic_file = codecs.open(args.topic_file, "r", "utf-8")

#################################
#new topic file processing begin#
#################################

new_topic_filename = args.topic_file.replace(".txt", "_new.txt")

if not os.path.exists(new_topic_filename):
	topics = []
	with codecs.open(args.topic_file, "r", "utf-8") as f:
		for line in f.readlines():
			line = line.strip()
			if line.startswith("w_h_top_words:['") and line.endswith("']"):
				topic_words = line.replace("w_h_top_words:['", "").replace("']", "").strip().split("', '")
				topic = " ".join(topic_words)
				topics.append(topic)

	with codecs.open(new_topic_filename, "w", "utf-8") as g:
		g.write("\n".join(topics))

#new input
topic_file = codecs.open(new_topic_filename, "r", "utf-8")

###############################
#new topic file processing end#
###############################

#global variables
#a dictionary that stores related topic words, e.g. { "space": set(["space", "earth", ...]), ... }
topic_word_rel = {}
unigram_list = [] #a list of unigrams (from topic words and candidates)
unigram_rev = {} #a reverse index of unigrams
word_count = {} #word counts (both single and pair)
corpus_partitions = [] #a list of the partitions of the corpus

#locks
wc_lock = threading.Lock()

####################
#call back function#
####################
def calcwcngram_complete(worker_wordcount):
	wc_lock.acquire()
	global num_comp
	global ord_count

	#update the wordcount from the worker
	for k, v in worker_wordcount.items():
		curr_v = 0
		if k in word_count:
			curr_v = word_count[k]
		curr_v += v
		word_count[k] = curr_v

	wc_lock.release()


##################
#worker functions#
##################
def convert_to_index(wordlist, unigram_rev):
	ids = []

	#for word in wordlist.split():
	for word in wordlist.strip().split():
		if word in unigram_rev:
			ids.append(unigram_rev[word])
		else:
			ids.append(0) 

	return ids

#update the word count of a given word
def update_word_count(word, worker_wordcount):
	count = 0
	if word in worker_wordcount:
		count = worker_wordcount[word]
	count += 1
	worker_wordcount[word] = count

#update the word count given a pair of words
def update_pair_word_count(w1, w2, topic_word_rel, worker_wordcount):
	if (w1 in topic_word_rel and w2 in topic_word_rel[w1]) or \
		(w2 in topic_word_rel and w1 in topic_word_rel[w2]):
		if w1 > w2:
			combined = w2 + "|" + w1
		else:
			combined = w1 + "|" + w2
		update_word_count(combined, worker_wordcount)

#given a sentence, find all ngrams (unigram or above)
def get_ngrams(words, topic_word_rel):
	all_ngrams = []
	ngram = []
	for i in range(0, len(words)):
		if (words[i] == 0):
			if len(ngram) > 0:
				all_ngrams.append(ngram)
				ngram = []
		else:
			ngram.append(unigram_list[words[i]-1])
	#append the last ngram
	if len(ngram) > 0:
		all_ngrams.append(ngram)
		ngram = []

	#permutation within ngrams
	ngrams_perm = []
	for ngram in all_ngrams:
		for i in range(1, len(ngram)+1):
			for j in range(0, len(ngram)-i+1):
				comb = [ item for item in ngram[j:j+i] ]
				ngrams_perm.append(' '.join(comb))

	#remove duplicates
	ngrams_perm = list(set(ngrams_perm))

	#only include ngrams that are found in topic words
	ngrams_final = []
	for ngram_perm in ngrams_perm:
		if ngram_perm in topic_word_rel:
			ngrams_final.append(ngram_perm)
		 
	return ngrams_final

#calculate word counts, given a list of words
def calc_word_count(words, topic_word_rel, unigram_list, worker_wordcount):

	ngrams = get_ngrams(words, topic_word_rel)

	for ngram in ngrams:
		if (ngram in topic_word_rel):
			update_word_count(ngram, worker_wordcount)

	for w1_id in range(0, len(ngrams)-1):
		for w2_id in range(w1_id+1, len(ngrams)):
			update_pair_word_count(ngrams[w1_id], ngrams[w2_id], topic_word_rel, worker_wordcount)

#primary worker function called by main
def calcwcngram(worker_num, window_size, corpus_file, topic_word_rel, unigram_list, unigram_rev):
	#now process the corpus file and sample the word counts
	line_num = 0
	worker_wordcount = {}
	total_windows = 0

	#sys.stderr.write("Worker " + str(worker_num) + " starts: " + str(time.time()) + "\n")
	for line in codecs.open(corpus_file, "r", "utf-8"):
		if corpus_lemmatization:
			#lemmatize corpus line
			line = " ".join([token.lemma_.lower() for token in nlp(line.strip())])

		#convert the line into a list of word indexes
		words = convert_to_index(line, unigram_rev)

		i=0
		doc_len = len(words)
		#number of windows
		if window_size != 0:
			num_windows = doc_len + window_size - 1
		else:
			num_windows = 1
		#update the global total number of windows
		total_windows += num_windows

		for tail_id in range(1, num_windows+1):
			if window_size != 0:
				head_id = tail_id - window_size
				if head_id < 0:
					head_id = 0
				words_in_window = words[head_id:tail_id]
			else:
				words_in_window = words

			calc_word_count(words_in_window, topic_word_rel, unigram_list, \
				worker_wordcount)

			i += 1

		line_num += 1

	#update the total windows seen for the worker
	worker_wordcount["!!<TOTAL_WINDOWS>!!"] = total_windows

	return worker_wordcount

################
#main functions#
################
#update the topic word - candidate words relation dictionary
def update_topic_word_rel(w1, w2):
	related_word_set = set([])
	if w1 in topic_word_rel:
		related_word_set = topic_word_rel[w1]
	if w2 != w1:
		related_word_set.add(w2)

	topic_word_rel[w1] = related_word_set
	
######
#main#
######

#get the partitions of the reference corpus
for f in os.listdir(args.ref_corpus_dir):
	if not f.startswith("."):
		corpus_partitions.append(args.ref_corpus_dir + "/" + f)

#process the topic file and get the topic word relation
unigram_set = set([]) #a set of all unigrams from the topic words
for line in topic_file:
	line = line.strip()
	topic_words = line.split()

	if topic_lemmatization:
		#topic lemmatization
		topic_words = [token.lemma_.lower() for token in nlp(" ".join(topic_words))]

	#update the unigram list and topic word relation
	for word1 in topic_words:
		#update the unigram first
		for word in word1.split(colloc_sep):
			unigram_set.add(word)

		#update the topic word relation
		for word2 in topic_words:
			if word1 != word2:
				#if it's collocation clean it so it's separated by spaces
				cleaned_word1 = " ".join(word1.split(colloc_sep))
				cleaned_word2 = " ".join(word2.split(colloc_sep))
				update_topic_word_rel(cleaned_word1, cleaned_word2)

#sort the unigrams and create a list and a reverse index
unigram_list = sorted(list(unigram_set))
unigram_rev = {}
unigram_id = 1
for unigram in unigram_list:
	unigram_rev[unigram] = unigram_id
	unigram_id += 1

#spawn multiple threads to process the corpus
po = Pool()
for i, cp in enumerate(corpus_partitions):
	sys.stderr.write("creating a thread for corpus partition " + cp + "\n")
	sys.stderr.flush()
	po.apply_async(calcwcngram, (i, window_size, cp, topic_word_rel, unigram_list, unigram_rev,), \
		callback=calcwcngram_complete)
po.close()
po.join()

#all done, print the word counts
with codecs.open(args.wordcount_file, "w", "utf-8") as f:
	for tuple in sorted(word_count.items()):
		f.write("%s|%s\n" % (str(tuple[0]), str(tuple[1])))

##############################################################################

ComputeCoherence(new_topic_filename, args.wordcount_file, args.coherence_file.replace(".txt", "_5.txt"), [5], "npmi", colloc_sep=colloc_sep, WTOTALKEY=WTOTALKEY)
ComputeCoherence(new_topic_filename, args.wordcount_file, args.coherence_file.replace(".txt", "_10.txt"), [10], "npmi", colloc_sep=colloc_sep, WTOTALKEY=WTOTALKEY)
ComputeCoherence(new_topic_filename, args.wordcount_file, args.coherence_file.replace(".txt", "_15.txt"), [15], "npmi", colloc_sep=colloc_sep, WTOTALKEY=WTOTALKEY)
ComputeCoherence(new_topic_filename, args.wordcount_file, args.coherence_file.replace(".txt", "_20.txt"), [20], "npmi", colloc_sep=colloc_sep, WTOTALKEY=WTOTALKEY)
ComputeCoherence(new_topic_filename, args.wordcount_file, args.coherence_file.replace(".txt", "_all.txt"), [5, 10, 15, 20], "npmi", colloc_sep=colloc_sep, WTOTALKEY=WTOTALKEY)