"""
Author:		 Jey Han Lau
Date:		   May 2013
"""

import argparse
import sys
import operator
import math
import codecs
import numpy as np
from collections import defaultdict


def ComputeCoherence(topic_file, wordcount_file, coherence_file, topns, metric,\
					colloc_sep="_", WTOTALKEY="!!<TOTAL_WINDOWS>!!"):
	
	#global variables
	window_total = 0 #total number of windows
	wordcount = {} #a dictionary of word counts, for single and pair words
	wordpos = {} #a dictionary of pos distribution

	###########
	#functions#
	###########

	#compute the association between two words
	def calc_assoc(word1, word2):
		combined1 = word1 + "|" + word2
		combined2 = word2 + "|" + word1

		combined_count = 0
		if combined1 in wordcount:
			combined_count = wordcount[combined1]
		elif combined2 in wordcount:
			combined_count = wordcount[combined2]
		w1_count = 0
		if word1 in wordcount:
			w1_count = wordcount[word1]
		w2_count = 0
		if word2 in wordcount:
			w2_count = wordcount[word2]

		if (metric == "pmi") or (metric == "npmi"):
			if w1_count == 0 or w2_count == 0 or combined_count == 0:
				result = 0.0
			else:
				result = math.log((float(combined_count)*float(window_total))/ \
					float(w1_count*w2_count), 10)
				if metric == "npmi":
					result = result / (-1.0*math.log(float(combined_count)/(window_total),10))

		elif metric == "lcp":
			if combined_count == 0:
				if w2_count != 0:
					result = math.log(float(w2_count)/window_total, 10)
				else:
					result = math.log(float(1.0)/window_total, 10)
			else:
				result = math.log((float(combined_count))/(float(w1_count)), 10)

		#if result < 0.0:
		#	import pdb; pdb.set_trace()
		
		return result

	#compute topic coherence given a list of topic words
	def calc_topic_coherence(topic_words):
		topic_assoc = []
		for w1_id in range(0, len(topic_words)-1):
			target_word = topic_words[w1_id]
			#remove the underscore and sub it with space if it's a collocation/bigram
			w1 = " ".join(target_word.split(colloc_sep))
			for w2_id in range(w1_id+1, len(topic_words)):
				topic_word = topic_words[w2_id]
				#remove the underscore and sub it with space if it's a collocation/bigram
				w2 = " ".join(topic_word.split(colloc_sep))
				if target_word != topic_word:
					topic_assoc.append(calc_assoc(w1, w2))
		
		#if float(sum(topic_assoc))/len(topic_assoc) < 0.0:
		#	import pdb; pdb.set_trace()

		return float(sum(topic_assoc))/len(topic_assoc)

	######
	#main#
	######

	#process the word count file(s)
	with codecs.open(wordcount_file, "r", "utf-8") as wc_file:
		for line in wc_file.readlines():
			line = line.strip()
			data = line.split("|")
			if len(data) == 2:
				wordcount[data[0]] = int(data[1])
			elif len(data) == 3:
				if data[0] < data[1]:
					key = data[0] + "|" + data[1]
				else:
					key = data[1] + "|" + data[0]
				wordcount[key] = int(data[2])
			else:
				#print "ERROR: wordcount format incorrect. Line =", line
				print("ERROR: wordcount format incorrect. Line =%s" % (line))
				raise SystemExit

	print("topns:  %s" % (str(topns)))

	#get the total number of windows
	if WTOTALKEY in wordcount:
		window_total = wordcount[WTOTALKEY]

	#read the topic file and compute the observed coherence
	topic_coherence = defaultdict(list) # {topicid: [tc]}
	topic_tw = {} #{topicid: topN_topicwords}
	with codecs.open(topic_file, "r", "utf-8") as to_file:
		for topic_id, line in enumerate(to_file.readlines()):
			topic_list = line.split()[:max(topns)]
			topic_tw[topic_id] = " ".join(topic_list)
			for n in topns:
				topic_coherence[topic_id].append(calc_topic_coherence(topic_list[:n]))


	#sort the topic coherence scores in terms of topic id
	tc_items = sorted(topic_coherence.items())
	mean_coherence_list = []
	with codecs.open(coherence_file, "w", "utf-8") as co_file:
		for item in tc_items:
			topic_words = topic_tw[item[0]].split()
			mean_coherence = np.mean(item[1])
			mean_coherence_list.append(mean_coherence)
			
			co_file.write("[%.2f] (" % mean_coherence),
			for i in item[1]:
				co_file.write("%.2f;" % i),
			co_file.write(")	%s\n" % topic_tw[item[0]])

		co_file.write("\n==========================================================================\n")
		co_file.write("Average Topic Coherence = %.3f\n" % (np.mean(mean_coherence_list)))
		co_file.write("Median Topic Coherence = %.3f\n" % (np.median(mean_coherence_list)))