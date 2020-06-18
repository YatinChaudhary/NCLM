## About

	This code consists of the implementations for the models proposed in the paper submitted: "Explainable and Discourse Topic-aware Neural Language Understanding"
	
	This code uses https://github.com/jhlau/topic_interpretability.git [1] github repository for topic coherence computations.


## Requirements

	NOTE: installation of correct dependencies and version ensure the correct working of code.

	Requires Python 3 (tested with `3.6.5`). The remaining dependencies can then be installed via:

        $ pip install -r requirements.txt
        $ python -c "import nltk; nltk.download('all')"


## Data preprocessing and format

	NOTE: we have already supplied "apnews" dataset in "datasets" directory. "bnc" and "imdb" datasets can be downloaded from the link below.

	1) Download the datasets from https://ibm.box.com/s/ls61p8ovc1y87w45oa02zink2zl7l6z4 [2]
	2) Extract the folder and transfer extracted "apnews", "bnc", "imdb" dataset directories to "datasets" directory
	3) Run these command-line scripts (running these scripts will take a while):
		a) "python preprocess_data_APNEWS.sh" 
		b) "python preprocess_data_BNC.sh" 
		c) "python preprocess_data_IMDB.sh"
	to prepare the datasets for langauge modeling task.

	Now "datasets" directory will contain different sub-directories for different datasets. Each sub-directory contains CSV format files for training, validation and test sets. The CSV files in each directory is named accordingly: 
		1) training_lstm_sents.csv    -->  training data input to the NLM component    (during NLM pretraining and joint training)
		2) validation_lstm_sents.csv  -->  validation data input to the NLM component  (during NLM pretraining and joint training)
		3) test_lstm_sents.csv        -->  test data input to the NLM component        (during NLM pretraining and joint training)
		
		4) training_nvdm_docs_minus_sents.csv     -->  training data input to the NTM component after removing sentence s from document d     (during joint training)
		5) validation_nvdm_docs_minus_sents.csv   -->  validation data input to the NTM component after removing sentence s from document d   (during joint training)
		6) test_nvdm_docs_minus_sents.csv         -->  test data input to the NTM component after removing sentence s from document d         (during joint training)
		
		7) training_nvdm_docs_non_replicated.csv    -->  training data input to the NTM component     (during NTM pretraining)
		8) validation_nvdm_docs_non_replicated.csv  -->  validation data input to the NTM component   (during NTM pretraining)
		9) test_nvdm_docs_non_replicated.csv        -->  test data input to the NTM component         (during NTM pretraining)

	Each sub-directory also contains two vocabulary files with 1 vocabulary token per line namely: 
		1) "vocab_nvdm.vocab"  -->  vocabulary file for NTM component
		2) "vocab_lstm.vocab"  -->  vocabulary file for NLM component

	NOTE:
		Each "...lstm_sents.csv" file contains one sentence in each line.
		Each "...nvdm_docs_minus_sents.csv" contains document after removing sentence present in corresponding "...lstm_sents.csv" file.
		Each "...nvdm_docs_non_replicated.csv" contains one document in each line without removing anything

		"./datasets/dummy_preprocessed_data"  -->  this directory shows above-mentioned files generated using "preprocess_data.py" script for a dummy input data. 

## How to use
	
	NOTE: You need to download the following pre-trained "word2vec" and "fasttext" embeddings before running the code:
		1) "word2vec"  -->  download "GoogleNews-vectors-negative300.bin" binary file using this link https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing [3]
		2) "fasttext"  -->  download the folder using this link https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip [4] and extract it to get "wiki.en.bin" binary file
	Place both of these pre-trained embeddings binary files in "./resources/pretrained_embeddings/" directory.
		

	The script "train_model_lstm.py" will train the Neural Composite Language Model (NCLM) framework and save it in a repository based on perplexity per word (PPL). It will also log all the training information in the same model folder. Here's how to use the script:
		
		$ python train_model_lstm.py --dataset  --model  --num-cores  --learning-rate --batch-size  --validation-bs  --test-bs  --log-every  --patience  --num-steps  --docnade-validation-ppl-freq  --docnade-validation-ir-freq  --lstm-validation-ppl-freq  --lstm-validation-ir-freq  --nvdm-validation-ppl-freq  --nvdm-validation-ir-freq  --test-ppl-freq  --test-ir-freq  --num-classes  --supervised  --hidden-size-TM  --hidden-size-LM  --hidden-size-LM-char  --deep-hidden-sizes  --activation  --TM-vocab-length  --LM-vocab-length  --rnn-char-vocab-length  --use-TM-for-ir  --use-lstm-for-ir  --use-combination-for-ir  --initialize-docnade  --initialize-nvdm  --initialize-rnn  --update-docnade-w  --update-rnn-w  --pretrain-LM  --pretrain-TM  --pretrain-epochs  --pretrain-patience  --combined-training  --combination-type  --docnade-loss-weight  --lstm-loss-weight  --use-bilm  --use-char-embeddings  --use-crf  --lstm-dropout-keep-prob  --tm-dropout-keep-prob  --common-space  --concat-proj-activation  --TM-type  --TM-lambda  --softmax-type  --alpha-uniqueness  --TM-uniqueness-loss  --beta-coherence  --topic-coherence-reg  --use-topic-embedding  --use-sent-topic-rep  --use-MOR  --prior-emb-for-topics  --use-k-topics  --use-k-topic-words  
		
		Description about each of the above command line arguments is provided in "train_model_lstm.py" file (at the bottom).
		
		Ready to run commandline scripts for all APNEWS, IMDB, BNC datasets have been included in this folder. There are 6 scripts in total:
			NOTE: Datasets have to be prepared before running these scripts.
		
			1) "train_APNEWS_new_TM_LM_small_NLM.sh"   --> script to run joint topic and language modeling for "APNEWS" dataset with (1-layer, 600 hidden units) LSTM-LM
			1) "train_APNEWS_new_TM_LM_large_NLM.sh"   --> script to run joint topic and language modeling for "APNEWS" dataset with (2-layer, 900 hidden units) LSTM-LM
			1) "train_BNC_new_TM_LM_small_NLM.sh"      --> script to run joint topic and language modeling for "BNC" dataset with (1-layer, 600 hidden units) LSTM-LM
			1) "train_BNC_new_TM_LM_large_NLM.sh"      --> script to run joint topic and language modeling for "BNC" dataset with (2-layer, 900 hidden units) LSTM-LM
			1) "train_IMDB_new_TM_LM_small_NLM.sh"     --> script to run joint topic and language modeling for "IMDB" dataset with (1-layer, 600 hidden units) LSTM-LM
			1) "train_IMDB_new_TM_LM_large_NLM.sh"     --> script to run joint topic and language modeling for "IMDB" dataset with (2-layer, 900 hidden units) LSTM-LM
		
		Hyperparameter settings for different configurations of NCLM framework are provided below:

			1) LTA-NLM:
				set argument "--use-topic-embedding" & "--prior-emb-for-topics" to False
				set global parameter "concat_topic_emb_and_prop" (in "./model/model_NVDM.py" file) to False
				set argument "--use-sent-topic-rep" to False
				
			2) ETA-NLM: 
				set argument "--use-topic-embedding" & "--prior-emb-for-topics" to True
				set global parameter "concat_topic_emb_and_prop" (in "./model/model_NVDM.py" file) to False
				set argument "--use-sent-topic-rep" to False
				
			3) LETA-NLM:
				set argument "--use-topic-embedding" & "--prior-emb-for-topics" to True
				set global parameter "concat_topic_emb_and_prop" (in "./model/model_NVDM.py" file) to True
				set argument "--use-sent-topic-rep" to False
				
			4) LTA-NLM +SDT:
				set argument "--use-topic-embedding" & "--prior-emb-for-topics" to False
				set global parameter "concat_topic_emb_and_prop" (in "./model/model_NVDM.py" file) to False
				set argument "--use-sent-topic-rep" to True
				
			5) ETA-NLM +SDT:
				set argument "--use-topic-embedding" & "--prior-emb-for-topics" to True
				set global parameter "concat_topic_emb_and_prop" (in "./model/model_NVDM.py" file) to False
				set argument "--use-sent-topic-rep" to True
				
			6) LETA-NLM +SDT:
				set argument "--use-topic-embedding" & "--prior-emb-for-topics" to True
				set global parameter "concat_topic_emb_and_prop" (in "./model/model_NVDM.py" file) to True
				set argument "--use-sent-topic-rep" to True


## Directory structure for experiments and datasets
	
	"model"  -->  this directory contains all saved models
	
	Based on the hyperparameter settings of an experiment, the following directory structure will be generated for the experiment.

	[Experiment directory]
		|
		|------ params.json         (file with hyperparameter settings saved in JSON format)
		|
		|------ ./model_ppl/        (directory containing model saved on the criteria of Perplexity (PPL))
		|
		|-------./topic_coherence/  (directory containing topic coherence [1] scores using internal (training dataset itself) and external corpus (wikipedia))
		|
		|------ ./logs/             (directory containing logs of model training and model reload)
					|
					|------ training_info.txt     (file containing negative log-likelihood loss and PPL evaluation score on "validation dataset" during training and "test dataset" after stopping criterion is acheived)
					|
					|------ TM_topics.txt         (file containing topics generated via NTM component)


## References

	[1] Lau, J.H., Newman, D. and Baldwin, T., 2014, April. Machine reading tea leaves: Automatically evaluating topic coherence and topic model quality. In Proceedings of the 14th Conference of the European Chapter of the Association for Computational Linguistics (pp. 530-539).
	[2] Lau, J.H., Baldwin, T. and Cohn, T., 2017. Topically driven neural language model. arXiv preprint arXiv:1704.08012.
	[3] Mikolov, T., Chen, K., Corrado, G. and Dean, J., 2013. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.
	[4] Bojanowski, P., Grave, E., Joulin, A. and Mikolov, T., 2017. Enriching word vectors with subword information. Transactions of the Association for Computational Linguistics, 5, pp.135-146.