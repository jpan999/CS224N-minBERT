from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *

NUM_TASKS = 3

class BertPals(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        #Encoder and decoder matrics project down to the smaller dimension
        self.aug_dense = nn.Linear(config.hidden_size, config.hidden_size_aug)
        self.aug_dense2 = nn.Linear(config.hidden_size_aug, config.hidden_size)
        self.attn = BertSelfAttentionPAL(config)
        self.config = config
        self.hidden_act_fn = F.gelu
        
    def forward(self, hidden_states, attention_mask=None):
        hidden_states_aug = self.aug_dense(hidden_states)
        hidden_states_aug = self.attn(hidden_states_aug, attention_mask)
        hidden_states = self.aug_dense2(hidden_states_aug)
        return hidden_states

# class BERTLowRank(nn.Module):
#     def __init__(self, config, extra_dim=None):
#         super(BERTLowRank, self).__init__()
#         # Encoder and decoder matrices project down to the smaller dimension
#         if config.extra_dim:
#             self.aug_dense = nn.Linear(config.hidden_size, config.extra_dim)
#             self.aug_dense2 = nn.Linear(config.extra_dim, config.hidden_size)
#         else:
#             self.aug_dense = nn.Linear(config.hidden_size, config.hidden_size_aug)
#             self.aug_dense2 = nn.Linear(config.hidden_size_aug, config.hidden_size)
#         self.config = config
#         self.hidden_act_fn = gelu

#     def forward(self, hidden_states, attention_mask=None):
#         hidden_states_aug = self.aug_dense(hidden_states)
#         hidden_states_aug = self.hidden_act_fn(hidden_states_aug)
#         hidden_states = self.aug_dense2(hidden_states_aug)
#         return hidden_states

class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this dropout is applied to normalized attention scores following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiply query and key 
    # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number 

    # normalize the scores
    # multiply the attention scores to the value and get back V'
    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]

    ### TODO
    
    #each k, q, v is projected to size [bs, num_attention_heads, seq_len, attention_head_size]
    bs, dummy, seq_len = key.shape[:3]
    
    #K^T
    key = key.transpose(2, 3)
    
    #QK^T
    score = query@key
    
    #dividing scaling factor, we get a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    score /= (self.attention_head_size**0.5)
    
    # attention mask to mask out the padding token scores
    score += attention_mask
    
    #applying softmax: input is of shape (s1, s2, s3, s4) then dim of softmax is 3
    sm = nn.Softmax(dim=3)
    score = sm(score)
    
    #TODO: do we need dropout
    #dropout applied to normalized attention scores
    score = self.dropout(score)
    
    # multiply the attention scores to the value and get back V'
    V_prime = score@value #[bs, num_attention_heads, seq_len, attention_head_size]
    
    #transpose V' to shape [bs, seq_len, num_attention_heads, attention_head_size]
    #get a contiguous copy after transpose
    V_prime = V_prime.transpose(1, 2).contiguous()
    
    #concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
    oup = V_prime.view(bs, seq_len, self.all_head_size) #[bs, seq_len, num_attention_heads * attention_head_size]
    
    return oup
    

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # calculate the multi-head attention 
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertSelfAttentionPAL(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size_aug / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size_aug, self.all_head_size)
    self.key = nn.Linear(config.hidden_size_aug, self.all_head_size)
    self.value = nn.Linear(config.hidden_size_aug, self.all_head_size)
    # this dropout is applied to normalized attention scores following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiply query and key 
    # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number 

    # normalize the scores
    # multiply the attention scores to the value and get back V'
    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]

    ### TODO
    
    #each k, q, v is projected to size [bs, num_attention_heads, seq_len, attention_head_size]
    bs, dummy, seq_len = key.shape[:3]
    
    #K^T
    key = key.transpose(2, 3)
    
    #QK^T
    score = query@key
    
    #dividing scaling factor, we get a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    score /= (self.attention_head_size**0.5)
    
    # attention mask to mask out the padding token scores
    score += attention_mask
    
    #applying softmax: input is of shape (s1, s2, s3, s4) then dim of softmax is 3
    sm = nn.Softmax(dim=3)
    score = sm(score)
    
    #TODO: do we need dropout
    #dropout applied to normalized attention scores
    score = self.dropout(score)
    
    # multiply the attention scores to the value and get back V'
    V_prime = score@value #[bs, num_attention_heads, seq_len, attention_head_size]
    
    #transpose V' to shape [bs, seq_len, num_attention_heads, attention_head_size]
    #get a contiguous copy after transpose
    V_prime = V_prime.transpose(1, 2).contiguous()
    
    #concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
    oup = V_prime.view(bs, seq_len, self.all_head_size) #[bs, seq_len, num_attention_heads * attention_head_size]
    
    return oup
    

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # calculate the multi-head attention 
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # multi-head attention
    self.self_attention = BertSelfAttention(config)
    # add-norm
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # another add-norm
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
    # print(config.extension_option)

    self.config = config
    # pal and lowrank extension
    multi = None
    self.multi_layers = None
    if config.extension_option == 'pal':
      multi = BertPals(config)
      # elif config.extension_option == 'lowrank':
      #   multi = BERTLowRank(config)
    self.multi_layers = nn.ModuleList([copy.deepcopy(multi) for _ in range(NUM_TASKS)])

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    this function is applied after the multi-head attention layer or the feed forward layer
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies to the output of each sub-layer, before it is added to the sub-layer input and normalized 
    ### TODO
    
    #BERT applies to the output of each sub-layer
    output = dense_layer(output)
    
    #dropout applies before adding to the sub-layer input and normalized
    output = dropout(output)
    
    #add to sub-layer input and normalized
    output = ln_layer(input + output)
    
    return output
    

  def forward(self, hidden_states, attention_mask, task_id=0):
    """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
    each block consists of 
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the input and output of the multi-head attention layer
    3. a feed forward layer
    4. a add-norm that takes the input and output of the feed forward layer
    """
    ### TODO
    
    #a multi-head attention layer (BertSelfAttention)
    oup1 = self.self_attention(hidden_states, attention_mask)
    
    #a add-norm that takes the input and output of the multi-head attention layer
    oup2 = self.add_norm(hidden_states, oup1, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
    
    #a feed forward layer
    oup3 = self.interm_dense(oup2)
    oup3 = self.interm_af(oup3)
    
    # print('task_id', task_id)
    #a add-norm that takes the input and output of the feed forward layer
    if self.config.extension_option == 'none':
      oup4 = self.add_norm(oup2, oup3, self.out_dense, self.out_dropout, self.out_layer_norm)
    else:
      # print('extension', config.extension_option)
      extra = self.multi_layers[task_id](hidden_states, attention_mask)
      oup4 = self.add_norm(oup2 + extra, oup3, self.out_dense, self.out_dropout, self.out_layer_norm)
    
    return oup4



class BertModel(BertPreTrainedModel):
  """
  the bert model returns the final embeddings for each token in a sentence
  it consists
  1. embedding (used in self.embed)
  2. a stack of n bert layers (used in self.encode)
  3. a linear transformation layer for [CLS] token (used in self.forward, as given)
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # bert encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # Get word embedding from self.word_embedding into input_embeds.
    inputs_embeds = None
    ### TODO
    inputs_embeds = self.word_embedding(input_ids)


    # Get position index and position embedding from self.pos_embedding into pos_embeds.
    pos_ids = self.position_ids[:, :seq_length]

    pos_embeds = None
    ### TODO
    pos_embeds = self.pos_embedding(pos_ids)


    # Get token type ids, since we are not consider token type, just a placeholder.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # Add three embeddings together; then apply embed_layer_norm and dropout and return.
    ### TODO
    embeds = inputs_embeds + pos_embeds + tk_type_embeds
    embeds = self.embed_layer_norm(embeds)
    embeds = self.embed_dropout(embeds)
    
    return embeds
    


  def encode(self, hidden_states, attention_mask, task_id):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # get the extended attention mask for self attention
    # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
    # non-padding tokens with 0 and padding tokens with a large negative number 
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # feed the encoding from the last bert_layer to the next
      hidden_states = layer_module(hidden_states, extended_attention_mask, task_id)

    return hidden_states

  def forward(self, input_ids, attention_mask, task_id=0):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask, task_id=task_id)

    # get cls token hidden state
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
