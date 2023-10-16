# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
import numpy as np
from src.expressions_transfer import *
from transformers import BertTokenizer

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

torch.cuda.set_device(0)
batch_size = 50
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
ori_path = './data/'
prefix = '23k_processed.json'



def seg(segmented_text):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    nums = []
    input_seq = []
    seg = segmented_text.strip().split(" ")

    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            nums.append(s[pos.start(): pos.end()])
            input_seq.append("NUM")
            if pos.end() < len(s):
                input_seq.append(s[pos.end():])
        else:
            input_seq.append(s)
    return input_seq

def get_train_test_fold(ori_path,prefix,data,pairs,group):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item,pair,g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g['group_num'])
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold

def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a/b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1])/100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num

import stanza
data = load_raw_data("data/Math_23K.json")
for i in data:
    question = i['original_text'].strip()
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    inputs = tokenizer(question, return_tensors="pt", add_special_tokens=False)
    token_list = [tokenizer.convert_ids_to_tokens(ii) for ii in inputs['input_ids']]
    
    question = ' '.join(token_list[0])
    idx = i['id']#.split('，')[-1]original_text
    #question = i['original_text']
    #question = question.replace('？', '').strip()
    #question = ' '.join().replace(' ','')
    print(question)
    #nlp = stanza.Pipeline(lang='zh', processors='tokenize,pos,constituency')#tokenize,
    nlp = stanza.Pipeline(lang='zh', processors='tokenize,pos,lemma,depparse',tokenize_pretokenized=True)
    doc = nlp(question)
    print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')

    for sent in doc.sentences:
        #assert len(sent.words) == len(seg(i['segmented_text']))
        adj_matrix = np.zeros((len(sent.words), len(sent.words)))
        for word in sent.words:
            if word.head > 0:
                adj_matrix[word.id - 1, word.head - 1] = 1
                adj_matrix[word.head - 1, word.id - 1] = 1
    print(adj_matrix)


    #for sentence in doc.sentences:
    #    print(sentence.constituency)


exit()








group_data = read_json("data/Math_23K_processed.json")





pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs
dd = {}
ddd = {}
for iii in pairs:
    op_list = ['+', '-', '*', '/']
    output_string = ''.join(iii[1])
    if iii[1][0] in op_list:
        if iii[1][0] not in dd:
            dd[iii[1][0]] = 1
        else:
            dd[iii[1][0]] += 1
    if iii[1][0] in op_list and iii[1][1] in op_list:
        if output_string[:2] not in dd:
            dd[output_string[:2]] = 1
        else:
            dd[output_string[:2]] += 1
    if iii[1][0] in op_list and iii[1][1] in op_list and iii[1][2] in op_list:
        if output_string[:3] not in ddd:
            ddd[output_string[:3]] = 1
        else:
            ddd[output_string[:3]] += 1
print(dd)
print(ddd)
exit()
#train_fold, test_fold, valid_fold = get_train_test_fold(ori_path,prefix,data,pairs,group_data)

train_fold, valid_fold, test_fold = get_train_test_fold(ori_path,prefix,data,pairs,group_data)

best_acc_fold = []

pairs_tested = test_fold
#pairs_trained = valid_fold
pairs_trained = train_fold

input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)
# Initialize models
encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                        n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

for epoch in range(n_epochs):

    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()
    for idx in range(len(input_lengths)):
        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx])
        loss_total += loss
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    if epoch % 10 == 0 or epoch > n_epochs - 5:
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                        merge, output_lang, test_batch[5], beam_size=beam_size)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
        torch.save(encoder.state_dict(), "models/encoder")
        torch.save(predict.state_dict(), "models/predict")
        torch.save(generate.state_dict(), "models/generate")
        torch.save(merge.state_dict(), "models/merge")


