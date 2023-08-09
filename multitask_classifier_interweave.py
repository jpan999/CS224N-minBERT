import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask


TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.linear_sent = torch.nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)
        self.linear_para1 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_para2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_sim1 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_sim2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        return pooled_output


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        pooled_output = self.forward(input_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        out = self.linear_sent(pooled_output)
        return out


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        pooled_output1 = self.forward(input_ids_1, attention_mask_1)
        pooled_output1 = self.dropout(pooled_output1)
        pooled_output1 = self.linear_para1(pooled_output1)
        pooled_output2 = self.forward(input_ids_2, attention_mask_2)
        pooled_output2 = self.dropout(pooled_output2)
        pooled_output2 = self.linear_para2(pooled_output2)
        cos_dist = torch.diagonal(torch.mm(pooled_output1, pooled_output2.t()))
        # cos_dist = self.cos(pooled_output1, pooled_output2)
        return cos_dist


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        # Update: it is no longer passed to sigmoid in evaluation.py
        ### TODO
        pooled_output1 = self.forward(input_ids_1, attention_mask_1)
        pooled_output1 = self.dropout(pooled_output1)
        pooled_output1 = self.linear_sim1(pooled_output1)
        pooled_output2 = self.forward(input_ids_2, attention_mask_2)
        pooled_output2 = self.dropout(pooled_output2)
        pooled_output2 = self.linear_sim2(pooled_output2)
        cos_dist = torch.diagonal(torch.mm(pooled_output1, pooled_output2.t()))
        return cos_dist




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


# Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    # sentiment classification data
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # sentence pair data
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    
    # sentence similarity data
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        
        train_loss_sst = 0
        train_loss_para = 0
        train_loss_sts = 0
        num_batches = 0
        for batch_sst, batch_para, batch_sts in zip(tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
            , tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
            , tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):

            # train on SST
            # print('SST batch')
            batch = batch_sst
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            loss1 = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            train_loss_sst += loss1.item()

            # train on Para
            # print('Para batch')
            batch = batch_para
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            loss2 = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1).float(), reduction='sum') / args.batch_size

            train_loss_para += loss2.item()

            # train on STS
            # print('STS batch')
            batch = batch_sts
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            loss3 = F.mse_loss(logits, b_labels.view(-1).float(), reduction='sum') / args.batch_size

            train_loss_sts += loss3.item()

            optimizer.zero_grad() 
            total_loss = loss1 + loss2 + loss3
            total_loss.backward()
            optimizer.step()
            num_batches += 1

        train_loss_sst = train_loss_sst / (num_batches)
        train_loss_para = train_loss_para / (num_batches)
        train_loss_sts = train_loss_sts / (num_batches)
        print(f"Epoch {epoch}: SST: train loss :: {train_loss_sst :.3f}\n" +
            f"Para train loss :: {train_loss_para :.3f}\n" +
            f"STS train loss :: {train_loss_sts :.3f}")

        # Print progress and save best model
        print("Train eval:")
        train_paraphrase_accuracy, _, _, train_sentiment_accuracy, _, _, train_sts_corr, _, _ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        print("Dev eval:")
        dev_paraphrase_accuracy, _, _, dev_sentiment_accuracy, _, _, dev_sts_corr, _, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        dev_acc = np.mean((dev_paraphrase_accuracy, dev_sentiment_accuracy, dev_sts_corr))
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        # print(f"Epoch {epoch}: SST train acc :: {train_sentiment_accuracy :.3f}, dev acc :: {dev_sentiment_accuracy :.3f}\n" + 
        #     f"Para train acc :: {train_paraphrase_accuracy}, dev acc :: {dev_paraphrase_accuracy}\n" + 
        #     f"STS train corr :: {train_sts_corr}, dev acc :: {dev_sts_corr}")


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    
    # pal and lowrank extension
    parser.add_argument("--extension_option", type=str, choices=('none', 'pal', 'lowrank'), default="none")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-{args.extension_option}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
