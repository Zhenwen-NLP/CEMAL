import random
import json
import copy
import re
import os
import pandas as pd
import nltk
try:
	nltk.download('punkt')
except:
	pass
import pdb
from utils.expressions_transfer import from_infix_to_prefix, compute_prefix_expression

PAD_token = 0

class Lang:
	"""
	class to save the vocab and two dict: the word->index and index->word
	"""
	def __init__(self):
		self.word2index = {}
		self.word2count = {}
		self.index2word = []
		self.n_words = 0  # Count word tokens
		self.num_start = 0

	def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
		for word in sentence:
			if re.search("N\d+|NUM|\d+", word):
				continue
			if word not in self.index2word:
				self.word2index[word] = self.n_words
				self.word2count[word] = 1
				self.index2word.append(word)
				self.n_words += 1
			else:
				self.word2count[word] += 1

	def trim(self, logger, min_count):  # trim words below a certain count threshold
		keep_words = []

		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)

		logger.debug('keep_words {} / {} = {}'.format(len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)))

		# Reinitialize dictionaries
		self.word2index = {}
		# self.word2count = {}
		self.index2word = []
		self.n_words = 0  # Count default tokens

		for word in keep_words:
			self.word2index[word] = self.n_words
			self.index2word.append(word)
			self.n_words += 1

	def build_input_lang(self, logger, trim_min_count):  # build the input lang vocab and dict
		if trim_min_count > 0:
			self.trim(logger, trim_min_count)
			self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
		else:
			self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
		self.word2index = {}
		self.n_words = len(self.index2word)
		for i, j in enumerate(self.index2word):
			self.word2index[j] = i

	def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
		self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
						  ["SOS", "UNK"]
		self.n_words = len(self.index2word)
		for i, j in enumerate(self.index2word):
			self.word2index[j] = i

	def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
		self.num_start = len(self.index2word)

		self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
		self.n_words = len(self.index2word)

		for i, j in enumerate(self.index2word):
			self.word2index[j] = i

def load_augmented_data(data_path, dataset = '', type_aug = '', size = 2):  # load the data to list(dict())
	if type_aug != '':
		train_path = os.path.join(data_path, dataset, type_aug)
	else:
		train_path = data_path
	train_df = pd.read_csv(train_path, header=None, sep='\|+',names=['Index', 'Question','Equation'], index_col = False, encoding= 'unicode_escape', dtype={'Index':str, 'Question':str, 'Equation':str})
	#shuffle and keep the first #size rows for each index
	train_df = train_df.sample(frac=1).groupby('Index').head(size)
	
	return train_df.to_dict('records')

def load_augmented_data_id(data_path, id_list, dataset = '', type_aug = ''):  # load the data to list(dict())
	if type_aug != '':
		train_path = os.path.join(data_path, dataset, type_aug)
	else:
		train_path = data_path
	train_df = pd.read_csv(train_path, header=None, sep='\|+',names=['Index', 'Question','Equation'], index_col = False, encoding= 'unicode_escape', dtype={'Index':str, 'Question':str, 'Equation':str})

	train_df = train_df[train_df['Index'].isin(id_list)]
	
	return train_df.to_dict('records')

def load_augmented_data_problemid(data_path, id_list, dataset = '', type_aug = ''):  # load the data to list(dict())
	if type_aug != '':
		train_path = os.path.join(data_path, dataset, type_aug)
	else:
		train_path = data_path
	train_df = pd.read_csv(train_path, header=None, sep='\|+',names=['Index', 'Question','Equation'], index_col = False, encoding= 'unicode_escape', dtype={'Index':str, 'Question':str, 'Equation':str})
	#print(train_df['Index'])
	train_df = train_df[train_df['Index'].isin(id_list)]
	
	return train_df.to_dict('records')

def retrive_augmented_data(data_path, problem_list, dataset = '', type_aug = '', size = 5):  # load the data to list(dict())
	if type_aug != '':
		train_path = os.path.join(data_path, dataset, type_aug)
	else:
		train_path = data_path
	train_df = pd.read_csv(train_path, header=None, sep='\|+',names=['Index', 'Question','Equation'], index_col = False, encoding= 'unicode_escape', dtype={'Index':str, 'Question':str, 'Equation':str})

	train_df = train_df[train_df['Index'].isin(problem_list)]

	#shuffle and only keep 4 rows for each problem
	train_df = train_df.sample(frac=1)
	train_df = train_df.groupby('Index').head(size)

	return train_df.to_dict('records')

def load_augmented_data_sizes(data_path, size = 1, dataset = '', type_aug = ''):  # load the data to list(dict())
	if type_aug != '':
		train_path = os.path.join(data_path, dataset, type_aug)
	else:
		train_path = data_path
	train_df = pd.read_csv(train_path, header=None, sep='\|+',names=['Index', 'Question','Equation'], index_col = False, encoding= 'unicode_escape', dtype={'Index':str, 'Question':str, 'Equation':str})

	#give each index a size
	train_df['size'] = train_df.groupby('Index').cumcount() + 1
	#shuffle the data
	train_df = train_df.sample(frac=1)

	#only keep the number of rows with the same index specified by propotion of its own size
	train_df = train_df.groupby('Index').apply(lambda x: x.sample(frac=size)).reset_index(drop=True)

	return train_df.to_dict('records')

def remove_brackets(x):
	y = x
	if x[0] == "(" and x[-1] == ")":
		x = x[1:-1]
		flag = True
		count = 0
		for s in x:
			if s == ")":
				count -= 1
				if count < 0:
					flag = False
					break
			elif s == "(":
				count += 1
		if flag:
			return x
	return y

def transfer_num_augmented(train_ls, pairs_trained, copy_nums, generate_nums):  # transfer num into "NUM"


	train_pairs = []
	for d in train_ls:
		try:
			problem_idx = int(d["Index"])
			original_problem = pairs_trained[problem_idx]
			
			# nums = []
			nums = None
			idxs = []
			ops = ['+','-','*','/','(',')','^']
			seg = d["Question"].split()
			replace_dict = {}
			num_idx = 0
			for idx, token in enumerate(seg):
				if token[0] == '$' and len(token) > 1:
					token = token[1:]
				if len(token) > 1:
					if token[0] == 'N' and token[1].isdigit():
						seg[idx] = 'NUM'
						idxs.append(idx)
						if token[1:1+len(str(num_idx))] != str(num_idx):
							replace_dict[token] = 'N' + str(num_idx)
						num_idx += 1

			equation = d["Equation"].replace('(','( ').replace(')',' )').replace('  ',' ').split()
			real_eq = []
			for eq_idx in range(len(equation)):
				if equation[eq_idx] in ops or equation[eq_idx][0] == 'N' or equation[eq_idx] in generate_nums:
					if len(equation[eq_idx]) < 4:
						if replace_dict != {} and equation[eq_idx] in replace_dict:
							real_eq.append(replace_dict[equation[eq_idx]])
						else:
							real_eq.append(equation[eq_idx])


			try:
				real_eq = from_infix_to_prefix((remove_brackets(real_eq)))
				temp_eq = []
				temp_idx = 1
				for ii in real_eq:
					if ii[0] == 'N':
						temp_eq.append(str(temp_idx))
						temp_idx += 1
					else:
						temp_eq.append(str(ii))
				if not compute_prefix_expression(temp_eq):
					#print(temp_eq)
					continue
			except:
				continue
			if len(original_problem[2]) == len(idxs):
				nums = original_problem[2]
				#print('Good')
			elif len(original_problem[2]) > len(idxs):
				nums = original_problem[2][:len(idxs)]
			else:
				nums = original_problem[2]
				for temp_num in range(len(idxs) - len(original_problem[2])):
					nums.append(str(temp_num+1))
			if idxs == []:
				continue
			assert len(seg) >= max(idxs)
			if copy_nums < len(nums):
					copy_nums = len(nums)

			train_pairs.append((seg, real_eq, nums, idxs, 0))
		except:
			continue
	return train_pairs, copy_nums


def transfer_num_augmented_during_training(train_ls, original_problem, copy_nums, generate_nums):  # transfer num into "NUM"


	train_pairs = []
	for d in train_ls:
		replace_dict = {}
		num_idx = 0
		# nums = []
		nums = None
		idxs = []
		ops = ['+','-','*','/','(',')','^','x']
		try:
			seg = d["Question"].split()
		except:
			continue
		for idx, token in enumerate(seg):
			if token[0] == '$' and len(token) > 1:
				token = token[1:]
			if len(token) > 1:
				if token[0] == 'N' and token[1].isdigit():
					seg[idx] = 'NUM'
					idxs.append(idx)
					if token[1:1+len(str(num_idx))] != str(num_idx):
						replace_dict[token] = 'N' + str(num_idx)
					num_idx += 1

		try:
			equation = d["Equation"].replace('(','( ').replace(')',' )').replace('  ',' ').split()
		except:
			continue
		
		real_eq = []
		for eq_idx in range(len(equation)):
			if equation[eq_idx] in ops or equation[eq_idx][0] == 'N' or equation[eq_idx] in generate_nums:
				if equation[eq_idx] == 'x':
					real_eq.append('*')
				if len(equation[eq_idx]) < 4:
					if replace_dict != {} and equation[eq_idx] in replace_dict:
						real_eq.append(replace_dict[equation[eq_idx]])
					else:
						real_eq.append(equation[eq_idx])
		try:
			real_eq = from_infix_to_prefix((remove_brackets(real_eq)))
			temp_eq = []
			temp_idx = 1
			for ii in real_eq:
				if ii[0] == 'N':
					temp_eq.append(str(temp_idx))
					temp_idx += 1
				else:
					temp_eq.append(str(ii))
			if not compute_prefix_expression(temp_eq):
				
				#print('question', d["Question"])
				#print('equation', d["Equation"])
				#print(temp_eq)
				continue
			#print(real_eq)
		except:
			continue
		if len(original_problem[4]) == len(idxs):
			nums = original_problem[4]
			#print('Good')
		elif len(original_problem[4]) > len(idxs):
			nums = original_problem[4][:len(idxs)]
		else:
			nums = original_problem[4]
			for temp_num in range(len(idxs) - len(original_problem[4])):
				nums.append(str(temp_num+1))
		if idxs == []:
			continue
		assert len(seg) >= max(idxs)
		if copy_nums < len(nums):
			continue

		train_pairs.append((seg, real_eq, nums, idxs, 0))
	return train_pairs, copy_nums

def load_raw_data(data_path, dataset, is_train = True):  # load the data to list(dict())
	train_ls = None
	if is_train:
		train_path = os.path.join(data_path, dataset, 'train.csv')
		train_df = pd.read_csv(train_path)
		train_df['id'] = train_df.index
		train_ls = train_df.to_dict('records')

	dev_path = os.path.join(data_path, dataset, 'test.csv')
	dev_df = pd.read_csv(dev_path)
	dev_df['id'] = dev_df.index
	dev_ls = dev_df.to_dict('records')

	return train_ls, dev_ls

# remove the superfluous brackets


def transfer_num(train_ls, dev_ls, chall=False):  # transfer num into "NUM"
	print("Transfer numbers...")
	dev_pairs = []
	generate_nums = []
	generate_nums_dict = {}
	copy_nums = 0

	if train_ls != None:
		train_pairs = []
		for d in train_ls:
			# nums = []
			nums = d['Numbers'].split()
			input_seq = []
			seg = nltk.word_tokenize(d["Question"].strip())
			equation = d["Equation"].split()

			numz = ['0','1','2','3','4','5','6','7','8','9']
			opz = ['+', '-', '*', '/']
			idxs = []
			for s in range(len(seg)):
				if len(seg[s]) >= 7 and seg[s][:6] == "number" and seg[s][6] in numz:
					input_seq.append("NUM")
					idxs.append(s)
				else:
					input_seq.append(seg[s])
			if copy_nums < len(nums):
				copy_nums = len(nums)

			out_seq = []
			for e1 in equation:
				if len(e1) >= 7 and e1[:6] == "number":
					out_seq.append('N'+e1[6:])
				elif e1 not in opz:
					generate_nums.append(e1)
					if e1 not in generate_nums_dict:
						generate_nums_dict[e1] = 1
					else:
						generate_nums_dict[e1] += 1
					out_seq.append(e1)
				else:
					out_seq.append(e1)

			train_pairs.append((input_seq, out_seq, nums, idxs, 1))
	else:
		train_pairs = None

	for d in dev_ls:
		# nums = []
		nums = d['Numbers'].split()
		input_seq = []
		seg = nltk.word_tokenize(d["Question"].strip())
		equation = d["Equation"].split()

		numz = ['0','1','2','3','4','5','6','7','8','9']
		opz = ['+', '-', '*', '/']
		idxs = []
		for s in range(len(seg)):
			if len(seg[s]) >= 7 and seg[s][:6] == "number" and seg[s][6] in numz:
				input_seq.append("NUM")
				idxs.append(s)
			else:
				input_seq.append(seg[s])
		if copy_nums < len(nums):
			copy_nums = len(nums)

		out_seq = []
		for e1 in equation:
			if len(e1) >= 7 and e1[:6] == "number":
				out_seq.append('N'+e1[6:])
			elif e1 not in opz:
				generate_nums.append(e1)
				if e1 not in generate_nums_dict:
					generate_nums_dict[e1] = 1
				else:
					generate_nums_dict[e1] += 1
				out_seq.append(e1)
			else:
				out_seq.append(e1)
		if chall:
			dev_pairs.append((input_seq, out_seq, nums, idxs, d['Type'], d['Variation Type'], d['Annotator'], d['Alternate']))
		else:
			dev_pairs.append((input_seq, out_seq, nums, idxs, 1))

	temp_g = []
	for g in generate_nums_dict:
		if generate_nums_dict[g] >= 5:
			temp_g.append(g)
	return train_pairs, dev_pairs, temp_g, copy_nums

# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
	res = []
	for word in sentence:
		if len(word) == 0:
			continue
		if word in lang.word2index:
			res.append(lang.word2index[word])
		else:
			res.append(lang.word2index["UNK"])
	if "EOS" in lang.index2word and not tree:
		res.append(lang.word2index["EOS"])
	return res

def sentence_from_indexes(lang, indexes):
	sent = []
	for ind in indexes:
		sent.append(lang.index2word[ind])
	return sent

def prepare_data(config, logger, pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, input_lang=None, output_lang=None, tree=False):
	if input_lang == None:
		input_lang = Lang()
	if output_lang == None:
		output_lang = Lang()

	test_pairs = []
	train_pairs = None

	if pairs_trained != None:
		train_pairs = []
		for pair in pairs_trained:
			if 'N' in pair[1] or '(' in pair[1] or ')' in pair[1]:
				continue
			if not tree:
				input_lang.add_sen_to_vocab(pair[0])
				output_lang.add_sen_to_vocab(pair[1])
			elif pair[3]:
				input_lang.add_sen_to_vocab(pair[0])
				output_lang.add_sen_to_vocab(pair[1])

	if config.embedding == 'bert' or config.embedding == 'roberta':
		for pair in pairs_tested:
			
			if not tree:
				input_lang.add_sen_to_vocab(pair[0])
			elif pair[3]:
				input_lang.add_sen_to_vocab(pair[0])

	if pairs_trained != None:
		input_lang.build_input_lang(logger, trim_min_count)
		if tree:
			output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
		else:
			output_lang.build_output_lang(generate_nums, copy_nums)

		for pair in pairs_trained:
			num_stack = []
			for word in pair[1]: # For each token in equation
				temp_num = []
				flag_not = True
				if word not in output_lang.index2word: # If token is not in output vocab
					flag_not = False
					for i, j in enumerate(pair[2]):
						if j == word:
							temp_num.append(i) # Append number list index of token not in output vocab

				if not flag_not and len(temp_num) != 0: # Equation has an unknown token and it is a number present in number list (could be default number with freq < 5)
					num_stack.append(temp_num)
				if not flag_not and len(temp_num) == 0: # Equation has an unknown token but it is not a number from number list
					num_stack.append([_ for _ in range(len(pair[2]))])

			num_stack.reverse()
			input_cell = indexes_from_sentence(input_lang, pair[0])
			output_cell = indexes_from_sentence(output_lang, pair[1], tree)
			break_flag = False
			if output_cell == []:
				continue
			#if 'Dan' in pair[0] and 'spent' in pair[0]:
			#	print(output_cell)
			#	print(max(output_cell) - output_lang.word2index['N0'] - 1)
			#	print(len(pair[3]))
				
			if max(output_cell) - output_lang.word2index['N0'] >= len(pair[3]):
				break_flag = True
			#if 'N0' not in pair[1]:
			#	break_flag = True
			for i in input_cell:
				if input_lang.index2word[i] == 'UNK':
					break_flag = True
			for i in output_cell:
				if output_lang.index2word[i] == 'UNK':
					break_flag = True
			if not break_flag:
				train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
									pair[2], pair[3], num_stack, pair[4]))
		print('Training Problems:', len(train_pairs))
	logger.debug('Indexed {} words in input language, {} words in output'.format(input_lang.n_words, output_lang.n_words))

	for pair in pairs_tested:
		num_stack = []
		for word in pair[1]:
			temp_num = []
			flag_not = True
			if word not in output_lang.index2word:
				flag_not = False
				for i, j in enumerate(pair[2]):
					if j == word:
						temp_num.append(i)

			if not flag_not and len(temp_num) != 0:
				num_stack.append(temp_num)
			if not flag_not and len(temp_num) == 0:
				num_stack.append([_ for _ in range(len(pair[2]))])

		num_stack.reverse()
		input_cell = indexes_from_sentence(input_lang, pair[0])
		output_cell = indexes_from_sentence(output_lang, pair[1], tree)
		break_flag = False
		for i in input_cell:
			if input_lang.index2word[i] == 'UNK':
				break_flag = True
		for i in output_cell:
			if output_lang.index2word[i] == 'UNK':
				break_flag = True
		if not break_flag:
			if config.challenge_disp:
				test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
							pair[2], pair[3], num_stack, pair[4], pair[5], pair[6], pair[7]))
			else:
				test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
							pair[2], pair[3], num_stack, pair[4]))
	print('Testing Problems:', len(test_pairs))
	return input_lang, output_lang, train_pairs, test_pairs

def prepare_data_augmented(config, logger, pairs_trained, trim_min_count, generate_nums, copy_nums, input_lang, output_lang, tree=False):


	train_pairs = None

	if pairs_trained != None:
		train_pairs = []
		for pair in pairs_trained:
			if 'N' in pair[1] or '(' in pair[1] or ')' in pair[1]:
				continue
			if not tree:
				input_lang.add_sen_to_vocab(pair[0])
				#output_lang.add_sen_to_vocab(pair[1])
			elif pair[3]:
				input_lang.add_sen_to_vocab(pair[0])
				#output_lang.add_sen_to_vocab(pair[1])


	if pairs_trained != None:

		for pair in pairs_trained:
			num_stack = []
			for word in pair[1]: # For each token in equation
				temp_num = []
				flag_not = True
				if word not in output_lang.index2word: # If token is not in output vocab
					flag_not = False
					for i, j in enumerate(pair[2]):
						if j == word:
							temp_num.append(i) # Append number list index of token not in output vocab

				if not flag_not and len(temp_num) != 0: # Equation has an unknown token and it is a number present in number list (could be default number with freq < 5)
					num_stack.append(temp_num)
				if not flag_not and len(temp_num) == 0: # Equation has an unknown token but it is not a number from number list
					num_stack.append([_ for _ in range(len(pair[2]))])

			num_stack.reverse()
			input_cell = indexes_from_sentence(input_lang, pair[0])
			output_cell = indexes_from_sentence(output_lang, pair[1], tree)
			break_flag = False
			if output_cell == []:
				continue
			#if 'Dan' in pair[0] and 'spent' in pair[0]:
			#	print(output_cell)
			#	print(max(output_cell) - output_lang.word2index['N0'] - 1)
			#	print(len(pair[3]))
				
			if max(output_cell) - output_lang.word2index['N0'] >= len(pair[3]):
				break_flag = True
			#if 'N0' not in pair[1]:
			#	break_flag = True
			for i in input_cell:
				if input_lang.index2word[i] == 'UNK':
					break_flag = True
			for i in output_cell:
				if output_lang.index2word[i] == 'UNK':
					break_flag = True
			if not break_flag:
				train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
									pair[2], pair[3], num_stack, pair[4]))
	return train_pairs

# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
	seq += [PAD_token for _ in range(max_length - seq_len)]
	return seq

# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size):
	pairs = copy.deepcopy(pairs_to_batch)
	random.shuffle(pairs)  # shuffle the pairs
	pos = 0
	input_lengths = []
	output_lengths = []
	nums_batches = []
	batches = []
	input_batches = []
	output_batches = []
	num_stack_batches = []  # save the num stack which
	num_pos_batches = []
	num_size_batches = []
	while pos + batch_size < len(pairs):
		batches.append(pairs[pos:pos+batch_size])
		pos += batch_size
	batches.append(pairs[pos:])

	for batch in batches:
		batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
		input_length = []
		output_length = []
		for _, i, _, j, _, _, _, _ in batch:
			input_length.append(i)
			output_length.append(j)
		input_lengths.append(input_length)
		output_lengths.append(output_length)
		input_len_max = input_length[0]
		output_len_max = max(output_length)
		input_batch = []
		output_batch = []
		num_batch = []
		num_stack_batch = []
		num_pos_batch = []
		num_size_batch = []
		for i, li, j, lj, num, num_pos, num_stack, _ in batch:
			num_batch.append(len(num))
			input_batch.append(pad_seq(i, li, input_len_max))
			output_batch.append(pad_seq(j, lj, output_len_max))
			num_stack_batch.append(num_stack)
			num_pos_batch.append(num_pos)
			num_size_batch.append(len(num_pos))
		input_batches.append(input_batch)
		nums_batches.append(num_batch)
		output_batches.append(output_batch)
		num_stack_batches.append(num_stack_batch)
		num_pos_batches.append(num_pos_batch)
		num_size_batches.append(num_size_batch)
	return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches

def get_num_stack(eq, output_lang, num_pos):
	num_stack = []
	for word in eq:
		temp_num = []
		flag_not = True
		if word not in output_lang.index2word:
			flag_not = False
			for i, j in enumerate(num_pos):
				if j == word:
					temp_num.append(i)
		if not flag_not and len(temp_num) != 0:
			num_stack.append(temp_num)
		if not flag_not and len(temp_num) == 0:
			num_stack.append([_ for _ in range(len(num_pos))])
	num_stack.reverse()
	return num_stack


def load_cv_data(data_path, dataset, is_train = True):  # load the data to list(dict())
	train_ls = None
	if is_train:
		train_path = os.path.join(data_path, dataset, 'train.csv')
		train_df = pd.read_csv(train_path)
		train_ls = train_df.to_dict('records')

	dev_path = os.path.join(data_path, dataset, 'dev.csv')
	dev_df = pd.read_csv(dev_path)
	dev_ls = dev_df.to_dict('records')

	return train_ls, dev_ls

