
import data_provider
import re
import numpy as np

from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp
import h5py



def get_bert_features():	
	with h5py.File('bert_embeddings/prompts.h5', 'r') as hq:
		prompt_features = hq['prompts'][:]
	with h5py.File('bert_embeddings/responses.h5', 'r') as hr:
		response_features = hr['responses'][:]
	with h5py.File('bert_embeddings/masks.h5', 'r') as hm:
		masks = hm['masks'][:]

	return np.array(prompt_features[:, :, :]), np.array(response_features[:, :, :]), np.array(masks[:, :])


def get_features(examples, conversation_length, embedding_size, verbose=True):

	glove_6b50d = nlp.embedding.create('glove', source='glove.6B.100d')
	vocab = nlp.Vocab(nlp.data.Counter(glove_6b50d.idx_to_token))
	vocab.set_embedding(glove_6b50d)

	max_conversation_length = conversation_length
	if verbose:
		print('max conversation length {}'.format(max_conversation_length))

	prompt_features = []
	response_features = []
	masks = []
	
	#temporary until save/load for embeddings is figured out
	#return np.zeros([len(examples), max_conversation_length, embedding_size]), np.zeros([len(examples), max_conversation_length, embedding_size]), np.zeros([len(examples), max_conversation_length])

	for c_i, conversation in enumerate(examples):
		if verbose:
			print('Getting features for example {}'.format(c_i))

		
		prompt_features.append(
			np.stack([get_vector_representation(conversation[i][0], vocab) if i < len(conversation) else np.zeros(embedding_size) 
				for i in range(max_conversation_length)])
		)
		
		response_features.append(
			np.stack([get_vector_representation(conversation[i][1], vocab) if i < len(conversation) else np.zeros(embedding_size) 
				for i in range(max_conversation_length)])
		)
		masks.append(
			np.array([1 if i < len(conversation) else 0 for i in range(max_conversation_length)])
		)
		

	

	return np.stack(prompt_features), np.stack(response_features), np.stack(masks)


def get_soa_features(examples, conversation_length, sequence_length, embedding_size, verbose=True):

	glove_6b50d = nlp.embedding.create('glove', source='glove.840B.300d')
	vocab = nlp.Vocab(nlp.data.Counter(glove_6b50d.idx_to_token))
	vocab.set_embedding(glove_6b50d)

	max_conversation_length = conversation_length
	if verbose:
		print('max conversation length {}'.format(max_conversation_length))

	prompt_features = []
	response_features = []
	masks = []
	
	#temporary until save/load for embeddings is figured out
	#return np.zeros([len(examples), max_conversation_length, embedding_size]), np.zeros([len(examples), max_conversation_length, embedding_size]), np.zeros([len(examples), max_conversation_length])

	for c_i, conversation in enumerate(examples):
		if verbose:
			print('Getting features for example {}'.format(c_i))

		
		response_features.append(
			np.stack([get_matrix_representation(conversation[i], vocab, sequence_length, embedding_size) if i < len(conversation) else np.zeros([sequence_length, embedding_size]) 
				for i in range(max_conversation_length)])
		)
		#masks.append(
		#	np.array([1 if i < len(conversation) else 0 for i in range(max_conversation_length)])
		#)
		

	

	return np.stack(response_features)


def get_redo_features(examples, conversation_length, sequence_length, embedding_size, verbose=True):
	glove_6b50d = nlp.embedding.create('glove', source='glove.840B.300d')
	vocab = nlp.Vocab(nlp.data.Counter(glove_6b50d.idx_to_token))
	vocab.set_embedding(glove_6b50d)

	max_conversation_length = conversation_length
	if verbose:
		print('max conversation length {}'.format(max_conversation_length))

	prompt_features = []
	response_features = []
	masks = []
	
	#temporary until save/load for embeddings is figured out
	#return np.zeros([len(examples), max_conversation_length, embedding_size]), np.zeros([len(examples), max_conversation_length, embedding_size]), np.zeros([len(examples), max_conversation_length])

	for c_i, conversation in enumerate(examples):
		if verbose:
			print('Getting features for example {}'.format(c_i))

		prompt_features.append(
			np.stack([get_matrix_representation(conversation[i][0], vocab, sequence_length, embedding_size) if i < len(conversation) else np.zeros([sequence_length, embedding_size]) 
				for i in range(max_conversation_length)])
		)
		
		response_features.append(
			np.stack([get_matrix_representation(conversation[i][1], vocab, sequence_length, embedding_size) if i < len(conversation) else np.zeros([sequence_length, embedding_size]) 
				for i in range(max_conversation_length)])
		)

	combined_features = np.zeros([2, max_conversation_length, sequence_length, embedding_size])
	return combined_features


def get_matrix_representation(text, vocab, sequence_length, embedding_size):
	tokens = preprocess_glove(text)
	representation = vocab.embedding[tokens].asnumpy()

	matrix = np.zeros([sequence_length, embedding_size])

	length = representation.shape[0]

	matrix[:length,:] = representation




	return matrix


def get_vector_representation(text, vocab):
		
	


	tokens = preprocess_glove(text)
	representation = np.average(
		vocab.embedding[tokens].asnumpy(),
		axis=0
	)

	return representation
				
def preprocess_glove(sentence):

	#GloVE doesn't take contractions

	contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

	def _get_contractions(contraction_dict):
		contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
		return contraction_dict, contraction_re

	contractions, contractions_re = _get_contractions(contraction_dict)

	def replace_contractions(text):
		def replace(match):
			return contractions[match.group(0)]
		
		return contractions_re.sub(replace, text)

	contractions_removed = replace_contractions(sentence)

	def simple_tokenize(source_str):
		return source_str.split(' ')

	tokens = simple_tokenize(contractions_removed)
	tokens = [token[:token.index('\'')] if '\'' in token else token for token in tokens]
	return tokens




def get_vocabulary_embeddings(examples):
	glove_6b50d = nlp.embedding.create('glove', source='glove.6B.100d')
	vocab = nlp.Vocab(nlp.data.Counter(glove_6b50d.idx_to_token))
	vocab.set_embedding(glove_6b50d)

	prompt_text = ' '.join([' '.join([' '.join(preprocess_glove(turn[0])) for turn in conversation]) for conversation in examples])
	respon_text = ' '.join([' '.join([' '.join(preprocess_glove(turn[1])) for turn in conversation]) for conversation in examples])
	tokens = (prompt_text + respon_text).split(' ')

	vocabulary = sorted(list(set(tokens)))
	print('Total vocabulary {}'.format(len(vocabulary)))


	for token in vocabulary:
		if token not in vocab:
			print(token)

	vocabulary = [token for token in vocabulary if token in vocab]
	print('Embeddable vocabulary {}'.format(len(vocabulary)))


