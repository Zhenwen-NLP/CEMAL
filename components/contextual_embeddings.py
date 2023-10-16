import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, AutoModel, AutoTokenizer
import pdb


class BertEncoder(nn.Module):
   def __init__(self, bert_model = 'bert-base-uncased',device = 'cuda:0 ', freeze_bert = False):
       super(BertEncoder, self).__init__()
       self.bert_layer = BertModel.from_pretrained(bert_model)
       self.tokenizer = BertTokenizer.from_pretrained(bert_model)
       self.device = device
      
       if freeze_bert:
           for p in self.bert_layer.parameters():
               p.requires_grad = False
      
   def bertify_input(self, sentences):
       '''
       Preprocess the input sentences using bert tokenizer and converts them to a torch tensor containing token ids


       '''
       #Tokenize the input sentences for feeding into BERT
       all_tokens  = [['[CLS]'] + self.tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]
      
       #Pad all the sentences to a maximum length
       input_lengths = [len(tokens) for tokens in all_tokens]
       max_length    = max(input_lengths)
       padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]


       #Convert tokens to token ids
       token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)


       #Obtain attention masks
       pad_token = self.tokenizer.convert_tokens_to_ids('[PAD]')
       attn_masks = (token_ids != pad_token).long()


       return token_ids, attn_masks, input_lengths


   def forward(self, sentences):
       '''
       Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
       '''
       #Preprocess sentences
       token_ids, attn_masks, input_lengths = self.bertify_input(sentences)


       #Feed through bert
       cont_reps = self.bert_layer(token_ids, attention_mask = attn_masks)['last_hidden_state']
       #for i in token_ids:
       #       print(self.tokenizer.convert_ids_to_tokens(i))


       num_pos_list = []
       for i in token_ids:
           num_pos = []
           for idx, token in enumerate(self.tokenizer.convert_ids_to_tokens(i)):
               if token == 'nu' and self.tokenizer.convert_ids_to_tokens(i)[idx+1] == '##m':
                   num_pos.append(idx)
           num_pos_list.append(num_pos)


       return cont_reps, input_lengths, num_pos_list


class RobertaEncoder(nn.Module):
   def __init__(self, roberta_model = 'roberta-base', device = 'cuda:0 ', freeze_roberta = False):
       super(RobertaEncoder, self).__init__()
       self.roberta_layer = RobertaModel.from_pretrained(roberta_model)
       self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
       self.device = device
      
       if freeze_roberta:
           for p in self.roberta_layer.parameters():
               p.requires_grad = False
      
   def robertify_input(self, sentences):
       '''
       Preprocess the input sentences using roberta tokenizer and converts them to a torch tensor containing token ids


       '''
       # Tokenize the input sentences for feeding into RoBERTa
       all_tokens  = [['<s>'] + self.tokenizer.tokenize(sentence) + ['</s>'] for sentence in sentences]
      
       # Pad all the sentences to a maximum length
       input_lengths = [len(tokens) for tokens in all_tokens]
       max_length    = max(input_lengths)
       padded_tokens = [tokens + ['<pad>' for _ in range(max_length - len(tokens))] for tokens in all_tokens]


       # Convert tokens to token ids
       token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)


       # Obtain attention masks
       pad_token = self.tokenizer.convert_tokens_to_ids('<pad>')
       attn_masks = (token_ids != pad_token).long()


       return token_ids, attn_masks, input_lengths


   def forward(self, sentences):
       '''
       Feed the batch of sentences to a RoBERTa encoder to obtain contextualized representations of each token
       '''
       # Preprocess sentences
       token_ids, attn_masks, input_lengths = self.robertify_input(sentences)


       # Feed through RoBERTa
       cont_reps = self.roberta_layer(token_ids, attention_mask = attn_masks)['last_hidden_state']

       num_pos_list = []
       for i in token_ids:
           num_pos = []
           for idx, token in enumerate(self.tokenizer.convert_ids_to_tokens(i)):
               if 'NUM' in token:
                   num_pos.append(idx)
           num_pos_list.append(num_pos)
          

       return cont_reps, input_lengths, num_pos_list



class DebertaEncoder(nn.Module):
   def __init__(self, deberta_model = 'roberta-base', device = 'cuda:0 ', freeze_deberta = False):
       super(DebertaEncoder, self).__init__()
       self.roberta_layer = AutoModel.from_pretrained(deberta_model)
       self.tokenizer = AutoTokenizer.from_pretrained(deberta_model)
       self.device = device
      
       if freeze_deberta:
           for p in self.roberta_layer.parameters():
               p.requires_grad = False
      
   def debertify_input(self, sentences):
       '''
       Preprocess the input sentences using roberta tokenizer and converts them to a torch tensor containing token ids


       '''
       # Tokenize the input sentences for feeding into RoBERTa

       all_tokens  = [['<s>'] + self.tokenizer.tokenize(sentence) + ['</s>'] for sentence in sentences]
      
       # Pad all the sentences to a maximum length
       input_lengths = [len(tokens) for tokens in all_tokens]
       max_length    = max(input_lengths)
       padded_tokens = [tokens + ['<pad>' for _ in range(max_length - len(tokens))] for tokens in all_tokens]


       # Convert tokens to token ids
       token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)


       # Obtain attention masks
       pad_token = self.tokenizer.convert_tokens_to_ids('<pad>')
       attn_masks = (token_ids != pad_token).long()


       return token_ids, attn_masks, input_lengths


   def forward(self, sentences):
       '''
       Feed the batch of sentences to a RoBERTa encoder to obtain contextualized representations of each token
       '''
       # Preprocess sentences
       token_ids, attn_masks, input_lengths = self.debertify_input(sentences)


       # Feed through RoBERTa
       cont_reps = self.roberta_layer(token_ids, attention_mask = attn_masks)['last_hidden_state']

       num_pos_list = []
       for i in token_ids:
           num_pos = []
           for idx, token in enumerate(self.tokenizer.convert_ids_to_tokens(i)):
               if 'NUM' in token:
                   num_pos.append(idx)
           num_pos_list.append(num_pos)
          

       return cont_reps, input_lengths, num_pos_list