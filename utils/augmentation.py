import pandas as pd
import random
import json
import copy
import re
from copy import deepcopy
import numpy as np
import nltk
import openai as ai



def load_raw_data(train_path, test_path, is_train = True):  # load the data to list(dict())
    train_ls = None
    if is_train:
        train_df = pd.read_csv(train_path, converters={'group_nums': eval})
        train_df['id'] = train_df.index
        train_ls = train_df.to_dict('records')

    dev_df = pd.read_csv(test_path, converters={'group_nums': eval})
    dev_df['id'] = dev_df.index
    dev_ls = dev_df.to_dict('records')

    return train_ls, dev_ls

def transfer_num_no_tokenize(train_ls, dev_ls, chall = False):  # transfer num into "NUM"
    print("Transfer numbers...")
    dev_pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0

    if train_ls != None:
        train_pairs = []
        for d in train_ls:
            # nums = []
            if 'Numbers' not in d.keys():
                continue
            nums = d['Numbers'].split()
            seg = nltk.word_tokenize(d["Question"].strip())
            equation = d["Equation"].split()
            
            input_seq = []
            

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
            continue_flag = False
            out_seq = []
            for e1 in equation:
                if len(e1) >= 7 and e1[:6] == "number":
                    out_seq.append('N'+e1[6:])
                elif e1 not in opz:
                    continue_flag = True
                    generate_nums.append(e1)
                    if e1 not in generate_nums_dict:
                        generate_nums_dict[e1] = 1
                    else:
                        generate_nums_dict[e1] += 1
                    out_seq.append(e1)
                else:
                    out_seq.append(e1)
            if continue_flag:
                continue
            train_pairs.append((input_seq, out_seq, nums, idxs, d['group_nums'], d['id']))
    else:
        train_pairs = None

    for d in dev_ls:
        if 'Numbers' not in d.keys():
            continue
        # nums = []
        nums = d['Numbers'].split()
        input_seq = []
        try:
            seg = nltk.word_tokenize(d["Question"].strip())
        except:
            pdb.set_trace()
        equation = d["Equation"].split()

        numz = ['0','1','2','3','4','5','6','7','8','9']
        opz = ['+', '-', '*', '/']
        idxs = []
        for s in range(len(seg)):
            if len(seg[s]) >= 7 and seg[s][:6] == "number" and seg[s][6] in numz:
                input_seq.append("N" + seg[s][6:])
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
        
        for iidx in range(len(input_seq)):
            if input_seq[iidx] == 'z':
                input_seq[iidx] = 'x'
            if input_seq[iidx] == 'NUM':
                input_seq[iidx] = 'z'

        if chall:
            dev_pairs.append((input_seq, out_seq, nums, idxs, d['group_nums'], d['Type'], d['Variation Type'], d['Annotator'], d['Alternate'], d['id']))
    
        else:
            dev_pairs.append((input_seq, out_seq, nums, idxs, d['group_nums'], d['id']))
        

    temp_g = []
    for g in generate_nums_dict:
        if generate_nums_dict[g] >= 100:
            temp_g.append(g)
    return train_pairs, dev_pairs, temp_g, copy_nums

def prefixToInfix(prefix):
    stack = []
     
    # read prefix in reverse order
    i = len(prefix) - 1
    while i >= 0:
        if not is_operand(prefix[i]):
             
            # symbol is operand
            stack.append(prefix[i])
            i -= 1
        else:
           
            # symbol is operator
            str = "( " + stack.pop() + ' ' + prefix[i] + ' ' + stack.pop() + " )"
            stack.append(str)
            i -= 1
     
    return stack.pop()

def is_operand(c):
    return c in ['+','-','*','/','^']

def process_question(question):
    num_idx = 0
    processed_question = []
    for aa in question:
        if aa == 'NUM':
            processed_question.append('N'+str(num_idx))
            num_idx += 1
        else:
            processed_question.append(aa)
    return ' '.join(processed_question)

def generation_type_1(src_idx, out_file_name, question, answer, n_generation = 1):
    prompt = """
    Generate a math word problem and its answer with a exactly same format as the previous one without explaination, using N0, N1, ... to represent numbers without other number representations.
    Variant Problem:"""

    ai.api_key = ''
    try:
        out_file = open(out_file_name,'a+')
    except:
        out_file = open(out_file_name,'w')
    question = process_question(question)
    answer = prefixToInfix(answer)
    full_prompt = 'Source Problem: ' + question + '\nAnswer: ' + answer + prompt
    #print('\n')
    #print(full_prompt)
    #print(answer)
    '''
    completions = ai.Completion.create(
            engine='text-davinci-003',  # Determines the quality, speed, and cost.
            temperature=0.9,            # Level of creativity in the response
            prompt=full_prompt,           # What the user typed in
            max_tokens=100,             # Maximum tokens in the prompt AND response
            top_p = 0.75,
            n=n_generation,                        # The number of completions to generate
            stop=None,                  # An optional setting to control response generation
        )
    '''
    completions = ai.ChatCompletion.create(
        model='gpt-3.5-turbo',  # Determines the quality, speed, and cost.
        temperature=0.9,            # Level of creativity in the response
        #prompt=full_prompt,           # What the user typed in 
        messages=[
        #{"role": "system", "content": "You are a math word problem generator."},
        {"role": "user", "content": full_prompt}],
        max_tokens=100,             # Maximum tokens in the prompt AND response
        n=n_generation,                        # The number of completions to generate
        stop=None,                  # An optional setting to control response generation
    )
    
    #print(completions.choices[0].text.replace('\n', '|'))
    #print(completions.choices[0].message['content'].replace('\n', '|'))
    for choice in completions.choices:
        #print(choice.text)
        #print(choice.message['content'])
        #out_file.write(str(src_idx) + '|' + choice.text.replace('\n', '|') + '\n')
        out_file.write(str(src_idx) + '|' + choice.message['content'].replace('\n', '|') + '\n')
    #out_file.write(str(src_idx) + '|' + completions.choices[0].message['content'].replace('\n', '|') + '\n')
    out_file.flush()

    return None

def generation_type_2(src_idx, out_file_name, question, answer, n_generation = 1):
    prompt = """
    Generate variant for math word problems, the answer for variant should be correct and different from the source, with a exactly same format as the source, contain not only operands but also operators, do not contain mod operator, and  using N0, N1, ... to represent numbers without other number representations.
    Source Problem: after a typhoon N0 trees in haley 's backyard died . if she had grown N1 trees initially how many more trees survived the typhoon than those that died ?
    Answer: N1 - N0 - N0
    Variant Problem: after a typhoon N0 trees in haley 's backyard died . if she had grown N1 trees initially how many trees does she have left ?
    Answer: N1 - N0
    Source Problem: an industrial machine made N0 shirts yesterday and N1 shirts today . it can make N2 shirts a minute . how many minutes did the machine work yesterday ?
    Answer: N0 / N2
    Variant Problem: an industrial machine made N0 shirts yesterday and N1 shirts today . it can make N2 shirts a minute . how many minutes did the machine work in all ? 
    Answer: ( N0 + N1 ) / N2
    Source Problem: N0 red peaches N1 yellow peaches and N2 green peaches are in the basket. how many peaches are in the basket? 
    Answer: N0 + N1 + N2
    Variant Problem: N0 red peaches N1 yellow peaches and N2 green peaches are in the basket. how many more red peaches than yellow peaches are in the basket? 
    Answer: N0 - N1
    """

    ai.api_key = ''
    try:
        out_file = open(out_file_name,'a+')
    except:
        out_file = open(out_file_name,'w')
    
    question = process_question(question)
    answer = prefixToInfix(answer)

    #print('\n')
    
    #print(answer)
    full_prompt = prompt + 'Source Problem: ' + question + '\nAnswer: ' + answer + '\nVariant Problem:'
    '''
    completions = ai.Completion.create(
            engine='text-davinci-003',  # Determines the quality, speed, and cost.
            temperature=0.9,            # Level of creativity in the response
            prompt=full_prompt,           # What the user typed in
            max_tokens=100,             # Maximum tokens in the prompt AND response
            top_p = 0.75,
            n=n_generation,                        # The number of completions to generate
            stop=None,                  # An optional setting to control response generation
        )
    '''
    completions = ai.ChatCompletion.create(
        model='gpt-3.5-turbo',  # Determines the quality, speed, and cost.
        temperature=0.9,            # Level of creativity in the response
        #prompt=full_prompt,           # What the user typed in 
        messages=[
        #{"role": "system", "content": "You are a math word problem generator."},
        {"role": "user", "content": full_prompt}],
        max_tokens=100,             # Maximum tokens in the prompt AND response
        n=n_generation,                        # The number of completions to generate
        stop=None,                  # An optional setting to control response generation
    )
    
    #print(completions.choices[0].text.replace('\n', '|'))
    for choice in completions.choices:
        #print(choice.text)
        #print(choice.message['content'])
        #out_file.write(str(src_idx) + '|' + choice.text.replace('\n', '|') + '\n')
        out_file.write(str(src_idx) + '|' + choice.message['content'].replace('\n', '|') + '\n')
    #out_file.write(str(src_idx) + '|' + completions.choices[0].message['content'].replace('\n', '|') + '\n')
    out_file.flush()
    return None
