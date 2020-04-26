import csv
import numpy as np
import re
from sklearn.model_selection import train_test_split



def shuffle_stratified_split(features, labels, train_size, dev_size, seed):
	traindev_indices, test_indices, y_traindev, y_test = train_test_split(
		features,
		labels,        
		train_size=train_size,
		stratify=labels,
		random_state=seed
	)

	train_indices, dev_indices = train_test_split(
		traindev_indices,         
		train_size=1-dev_size,
		#train_size=train_size,
		stratify=y_traindev,
		random_state=seed
	)

	return train_indices, dev_indices, test_indices


def get_batches(data, batch_size, num_epochs, shuffle):
	"""
	Generates a batch iterator for a dataset.
	"""
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	print(num_batches_per_epoch)
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = [data[shuffle_index] for shuffle_index in shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
#			if indices_only:
#				sample = shuffled_data[start_index:end_index]
#				yield [[s[0], np.array([s[1], s[2]])] for s in sample]
#			else:
			yield shuffled_data[start_index:end_index]


def convert_to_one_hot(labels):
	return np.stack([np.array([1,0]) if label == 0 else np.array([0,1]) for label in labels])

def generate_data(verbose=True, return_indices=False):
	path_to_data = "Data/"

	features = []

	spkr_idx = 2
	utt_idx = 3

	sum_turns = 0
	sum_tokens = 0

	num_convos = 0

	turns = []

	strat_participants = None
	
	set_file = open('Data/participant_indices.txt', 'r')
	
        
        
	strat_participants = [int(s) for s in set_file.readlines()[0].split(',')]    
	print(len(strat_participants))
	
	for i in strat_participants:
		try:
			utterances = list(csv.reader(open('{}{}_TRANSCRIPT.csv'.format(path_to_data, i), 'r'), delimiter='\t'))
		except IOError as e:
			if verbose:
				print(e)
				print('Could not open {}_TRANSCRIPT.csv'.format(i))
			continue

		current_speaker = utterances[1][spkr_idx]
		current_turn = [current_speaker, '', i]

		turns_per = 0
		num_convos += 1

		try:
			for u in utterances[1:]:
				if u[spkr_idx] != current_speaker:
					# print('utt by {}: {}'.format(current_speaker, current_turn[1]))
					turns_per += 1
					sum_tokens += len(current_turn[1].split(' '))
					turns.append(current_turn)

					current_speaker = u[spkr_idx]
					current_turn = [current_speaker, u[utt_idx], i]
				else:         
					#continue appending to turn       
					current_turn[1] = '{} {}'.format(current_turn[1], u[utt_idx])
		except IndexError:
			if verbose:
				print('Index error. Moving on.')

		sum_turns += turns_per

	num_turns = len(turns)
	if verbose:
		print('number of conversations: {}'.format(num_convos))
		print('average turns per conversation: {}'.format(num_turns / num_convos))
		print('average tokens per turn: {}'.format(sum_tokens / num_turns))

	# group speaker/respondant turns into one group
	qa_pairs = []
	for i in range(num_turns):
		if i + 1 < num_turns and len(turns[i]) == 3 and len(turns[i + 1]) == 3 and \
				turns[i][0] == "Ellie" and turns[i + 1][0] == "Participant":

			question = turns[i][1]
			p = re.compile('\(.+\)')
			parenthetical = p.search(question)
			if parenthetical is not None:
				question = parenthetical.group(0).replace('(', '').replace(')', '')


			pair = [question, turns[i + 1][1], turns[i][2]]

			if pair[2] in strat_participants:
				qa_pairs.append(pair)

	if verbose:
		print('num qa pairs: {}'.format(len(qa_pairs)))

	# Retrieve labels
	dev_labels = list(csv.reader(open('Data/dev_split_Depression_AVEC2017.csv', 'r'), delimiter=','))
	train_labels = list(csv.reader(open('Data/train_split_Depression_AVEC2017.csv', 'r'), delimiter=','))
	additional_labels = list(csv.reader(open('Data/full_test_split.csv', 'r'), delimiter=','))

	all_labelsets = dev_labels[1:]
	all_labelsets.extend(train_labels[1:])
	all_labelsets.extend([a for a in additional_labels[1:] if len(a) > 0])

	examples = []
	example_refs = []
	

	regression_labels = []
	classification_labels = []

	# only include examples that have labels, group by participant
	for labelset in all_labelsets:
		part_turns = [q for q in qa_pairs if str(q[2]) == str(labelset[0])]
		if (len(part_turns) > 0):
			classification_labels.append(labelset[1])
			examples.append([[q[0], q[1]] for q in part_turns])
			example_refs.append(part_turns[0][2])


	if verbose:
		print('Percent depressed: {}'.format(sum([int(c) for c in classification_labels])/len(classification_labels)))
	
	print('average turns per conversation: {}'.format(num_turns / num_convos))
	print('max turns per conversation: {}'.format(max([len(example) for example in examples])))

	if return_indices:
		return examples, [int(c) for c in classification_labels], example_refs
	else:	
		return examples, [int(c) for c in classification_labels]


def generate_sentence_data(verbose=True, return_indices=False):
	path_to_data = "Data/"

	features = []

	spkr_idx = 2
	utt_idx = 3

	sum_turns = 0
	sum_tokens = 0

	num_convos = 0

	strat_participants = None
	
	set_file = open('Data/participant_indices.txt', 'r')
	strat_participants = [int(s) for s in set_file.readlines()[0].split(',')]    

	examples = []

	for i in strat_participants:
		turns = []
		try:
			utterances = list(csv.reader(open('{}{}_TRANSCRIPT.csv'.format(path_to_data, i), 'r'), delimiter='\t'))
		except IOError as e:
			if verbose:
				print(e)
				print('Could not open {}_TRANSCRIPT.csv'.format(i))
			continue

		turns_per = 0
		num_convos += 1


		
		try:
			for u in utterances[1:]:

				if u[spkr_idx] != "Participant":
					continue

				else:										
					utterance = (' '.join([token for token in u[utt_idx].split(' ') if token.replace("'", "").isalpha()])).strip()

					if len(utterance) < 1:
						continue

					current_turn = utterance
					turns_per +=1
					sum_tokens += len(current_turn.split(' '))
					turns.append(current_turn)
	
		except IndexError:
			if verbose:
				print('Index error. Moving on.')

		examples.append([i, turns])

		sum_turns += turns_per


	# Retrieve labels
	dev_labels = list(csv.reader(open('Data/dev_split_Depression_AVEC2017.csv', 'r'), delimiter=','))
	train_labels = list(csv.reader(open('Data/train_split_Depression_AVEC2017.csv', 'r'), delimiter=','))
	additional_labels = list(csv.reader(open('Data/full_test_split.csv', 'r'), delimiter=','))

	all_labelsets = dev_labels[1:]
	all_labelsets.extend(train_labels[1:])
	all_labelsets.extend([a for a in additional_labels[1:] if len(a) > 0])

	
	example_refs = []
	

	regression_labels = []
	classification_labels = []

	final_examples = []

	# only include examples that have labels, group by participant
	for labelset in all_labelsets:

		match = [example for example in examples if str(example[0]) == str(labelset[0])]
		if (len(match) > 0):
			example = match[0][1]
			classification_labels.append(labelset[1])
			final_examples.append(example)
			example_refs.append(labelset[0])

	examples = final_examples

	if verbose:
		print('Percent depressed: {}'.format(sum([int(c) for c in classification_labels])/len(classification_labels)))
	
	print('average turns per conversation: {}'.format(sum_turns / num_convos))
	print('max turns per conversation: {}'.format(max([len(example) for example in examples])))
	print('max tokens in conversation: {}'.format(max([max([len(turn.split(' ')) for turn in example]) for example in examples])))

	if return_indices:
		return examples, [int(c) for c in classification_labels], example_refs
	else:	
		return examples, [int(c) for c in classification_labels]
