import numpy as np
import pandas as pd
import torch
import time
import joblib
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, f1_score, recall_score, precision_score, 
                             precision_recall_curve, confusion_matrix, matthews_corrcoef)
import transformers
from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from azureml.core import Workspace, Run, Dataset
import matplotlib.pyplot as plt

# paramters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_path',
    type=str,
    help='Path to the training data'
)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=3e-5,
    help='Learning rate for SGD'
)
parser.add_argument(
    '--batch_size', 
    type=int, 
    dest='batch_size', 
    default=8
)
parser.add_argument(
    '--adam_epsilon', 
    type=float, 
    dest='adam_epsilon', 
    default=1e-8
)
parser.add_argument(
    '--num_epochs', 
    type=int, 
    dest='num_epochs', 
    default=3)

args = parser.parse_args()

batch_size = args.batch_size
learning_rate = args.learning_rate
adam_epsilon = args.adam_epsilon
num_epochs = args.num_epochs

print("Arguments: ", (batch_size, learning_rate, adam_epsilon, num_epochs))

# run = Run.get_context()

print("===== DATA =====")
print("DATA PATH: " + args.data_path)
print("LIST FILES IN DATA PATH...")
print(os.listdir(args.data_path))
print("================")

# ws = run.experiment.workspace

# # get the input dataset by ID
# dataset = Dataset.get_by_id(ws, id=args.data_path)

# # load the TabularDataset to pandas DataFrame
# df = dataset.to_pandas_dataframe()

df = pd.read_csv(args.data_path+'/newdatasetwithcoviddata.csv')
df.dropna(inplace=True)
df = df.sample(50000)

train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['label'], 
                                                                    random_state=2018, 
                                                                    test_size=0.4, 
                                                                    stratify=df['label'])

# we will use temp_text and temp_labels to create validation and test set
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=temp_labels)

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

## Tokenization

max_seq_len = 25

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)


## Convert Integer Sequences to Tensors

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

## Create DataLoaders

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# #define a batch size
# batch_size = 32

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)


## Freeze BERT Parameters

# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False


## Define Model Architecture

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      
      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)
      
      # apply softmax activation
      x = self.softmax(x)

      return x

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

# optimizer from hugging face transformers

# define the optimizer
optimizer = AdamW(model.parameters(), lr = learning_rate)

# adam_epsilon, num_epochs

## Find Class Weights

#compute the class weights
class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)

# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)

# loss function
cross_entropy  = nn.NLLLoss(weight=weights) 

## Fine Tune BERT

# function to train the model
def train():
  
  model.train()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  for step,batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]
 
    sent_id, mask, labels = batch

    # clear previously calculated gradients 
    model.zero_grad()        

    # get model predictions for the current batch
    preds = model(sent_id, mask)

    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)

    # add on to the total loss
    total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
    loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update parameters
    optimizer.step()

    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds

# function for evaluating the model
def evaluate():
  
  print("\nEvaluating...")
  
  # deactivate dropout layers
  model.eval()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save the model predictions
  total_preds = []

  # iterate over batches
  for step,batch in enumerate(val_dataloader):
    
    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
      # elapsed = format_time(time.time() - t0)
            
      # Report progress.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    # push the batch to gpu
    batch = [t.to(device) for t in batch]

    sent_id, mask, labels = batch

    # deactivate autograd
    with torch.no_grad():
      
      # model predictions
      preds = model(sent_id, mask)

      # compute the validation loss between actual and predicted values
      loss = cross_entropy(preds,labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()

      total_preds.append(preds)

  # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

out_dir = './outputs'

#for each epoch
for epoch in range(num_epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, num_epochs))
    
    #train model
    train_loss, _ = train()
    
    #evaluate model
    valid_loss, _ = evaluate()
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), out_dir+'/saved_weights.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')


## Save losses

with open(out_dir + '/train_losses.pkl', 'wb') as f:
    joblib.dump(train_losses, f)
with open(out_dir + '/val_losses.pkl', 'wb') as f:
    joblib.dump(valid_losses, f)

plt.plot(range(len(train_losses)), train_losses, label='training loss')
plt.plot(range(len(valid_losses)), valid_losses, label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(out_dir + '/losses.png')

## Load Saved Model

#load weights of best model
model.load_state_dict(torch.load(out_dir+'/saved_weights.pt'))

## Get Predictions for Test Data

# get predictions for test data
with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()

# model's performance
precision_, recall_, proba = precision_recall_curve(test_y, preds[:, -1])
preds = np.argmax(preds, axis = 1)

#plot precision-recall curve
plt.plot(recall_, precision_, marker='.', label='BERT-model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig(out_dir + '/precision-reall-curve.png')

# optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
# preds = [1 if i >= optimal_proba_cutoff else 0 for i in preds[:, -1]]

mcc = matthews_corrcoef(test_y, preds)
tn, fp, fn, tp = confusion_matrix(test_y, preds).ravel()
precision = precision_score(test_y, preds)
recall = recall_score(test_y, preds)
f1 = f1_score(test_y, preds, average='weighted')
cm = confusion_matrix(test_y, preds)

print("")
print("Matthews Corr Coef:", mcc)
print("Precision:", precision)
print("Recall:", recall)
print("f-1 score:", f1)
print("confusion Matrix:", cm)
print("")
print(classification_report(test_y, preds))

# from __future__ import absolute_import, division, print_function

# import glob
# import logging
# import os
# import random
# import json
# import numpy as np
# import torch
# from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
# import random
# from torch.utils.data.distributed import DistributedSampler
# from tqdm import tqdm_notebook, trange
# from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
#                                   XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
#                                   XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
#                                   RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
# from pytorch_transformers import AdamW, WarmupLinearSchedule
# from sklearn.metrics import (mean_squared_error, matthews_corrcoef, confusion_matrix, f1_score, 
# 							 precision_score, recall_score, precision_recall_curve, plot_precision_recall_curve)
# from scipy.stats import pearsonr
# from utils import (convert_examples_to_features, output_modes, processors)

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def load_and_cache_examples(task, tokenizer, evaluate=False, undersample_scale_factor=0.01):
#     processor = processors[task]()
#     output_mode = args['output_mode']
    
#     mode = 'dev' if evaluate else 'train'
#     cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")
    
#     if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
#         logger.info("Loading features from cached file %s", cached_features_file)
#         features = torch.load(cached_features_file)
               
#     else:
#         logger.info("Creating features from dataset file at %s", args['data_dir'])
#         label_list = processor.get_labels()
#         examples = processor.get_dev_examples(args['data_dir']) if evaluate else processor.get_train_examples(args['data_dir'])
#         print(len(examples))
#         examples  = [example for example in examples if np.random.rand() < undersample_scale_factor]
#         print(len(examples))
        
#         features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode,
#             cls_token_at_end=bool(args['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
#             cls_token=tokenizer.cls_token,
#             sep_token=tokenizer.sep_token,
#             cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
#             pad_on_left=bool(args['model_type'] in ['xlnet']),                 # pad on the left for xlnet
#             pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0,
#             process_count=2)
        
#         logger.info("Saving features into cached file %s", cached_features_file)
#         torch.save(features, cached_features_file)
        
#     all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
#     all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
#     if output_mode == "classification":
#         all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
#     elif output_mode == "regression":
#         all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

#     dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
#     return dataset

# def train(train_dataset, model, tokenizer):
#     train_sampler = RandomSampler(train_dataset)
#     train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])
    
#     t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']
    
#     no_decay = ['bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
#         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
#     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)
    
#     if args['fp16']:
#         try:
#             from apex import amp
#         except ImportError:
#             raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
#         model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])
        
#     logger.info("***** Running training *****")
#     logger.info("  Num examples = %d", len(train_dataset))
#     logger.info("  Num Epochs = %d", args['num_train_epochs'])
#     logger.info("  Total train batch size  = %d", args['train_batch_size'])
#     logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
#     logger.info("  Total optimization steps = %d", t_total)

#     global_step = 0
#     tr_loss, logging_loss = 0.0, 0.0
#     model.zero_grad()
#     train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")
    
#     for _ in train_iterator:
#         epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")
#         for step, batch in enumerate(epoch_iterator):
#             model.train()
#             batch = tuple(t.to(device) for t in batch)
#             inputs = {'input_ids':      batch[0],
#                       'attention_mask': batch[1],
#                       'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
#                       'labels':         batch[3]}
#             outputs = model(**inputs)
#             loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
#             print("\r%f" % loss, end='')

#             if args['gradient_accumulation_steps'] > 1:
#                 loss = loss / args['gradient_accumulation_steps']

#             if args['fp16']:
#                 with amp.scale_loss(loss, optimizer) as scaled_loss:
#                     scaled_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                
#             else:
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

#             tr_loss += loss.item()
#             if (step + 1) % args['gradient_accumulation_steps'] == 0:
#                 scheduler.step()  # Update learning rate schedule
#                 optimizer.step()
#                 model.zero_grad()
#                 global_step += 1

#                 if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
#                     # Log metrics
#                     if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
#                         results = evaluate(model, tokenizer)

#                     logging_loss = tr_loss

#                 if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
#                     # Save model checkpoint
#                     output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
#                     if not os.path.exists(output_dir):
#                         os.makedirs(output_dir)
#                     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
#                     model_to_save.save_pretrained(output_dir)
#                     logger.info("Saving model checkpoint to %s", output_dir)


#     return global_step, tr_loss / global_step

# def get_mismatched(labels, preds):
#     mismatched = labels != preds
#     examples = processor.get_dev_examples(args['data_dir'])
#     wrong = [i for (i, v) in zip(examples, mismatched) if v]
    
#     return wrong

# def get_eval_report(labels, preds):
#     mcc = matthews_corrcoef(labels, preds)
#     tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
#     precision = precision_score(labels, preds)
#     recall = recall_score(labels, preds)
#     f1 = f1_score(labels, preds, average='weighted')
#     return {
#         "mcc": mcc,
#         "f1":f1,
#         "precision":precision,
#         "recall":recall,
#         "tp": tp,
#         "tn": tn,
#         "fp": fp,
#         "fn": fn
#     }, get_mismatched(labels, preds)

# def compute_metrics(task_name, preds, labels):
#     assert len(preds) == len(labels)
#     return get_eval_report(labels, preds)

# def evaluate(model, tokenizer, prefix=""):
#     # Loop to handle MNLI double evaluation (matched, mis-matched)
#     eval_output_dir = args['output_dir']

#     results = {}
#     EVAL_TASK = args['task_name']

#     eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=True)
#     if not os.path.exists(eval_output_dir):
#         os.makedirs(eval_output_dir)


#     eval_sampler = SequentialSampler(eval_dataset)
#     eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

#     # Eval!
#     logger.info("***** Running evaluation {} *****".format(prefix))
#     logger.info("  Num examples = %d", len(eval_dataset))
#     logger.info("  Batch size = %d", args['eval_batch_size'])
#     eval_loss = 0.0
#     nb_eval_steps = 0
#     preds = None
#     out_label_ids = None
#     for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
#         model.eval()
#         batch = tuple(t.to(device) for t in batch)

#         with torch.no_grad():
#             inputs = {'input_ids':      batch[0],
#                       'attention_mask': batch[1],
#                       'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
#                       'labels':         batch[3]}
#             outputs = model(**inputs)
#             tmp_eval_loss, logits = outputs[:2]

#             eval_loss += tmp_eval_loss.mean().item()
#         nb_eval_steps += 1
#         if preds is None:
#             preds = logits.detach().cpu().numpy()
#             out_label_ids = inputs['labels'].detach().cpu().numpy()
#         else:
#             preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
#             out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

#     eval_loss = eval_loss / nb_eval_steps
#     if args['output_mode'] == "classification":
#         # preds = np.argmax(preds, axis=1)
#         precision_, recall_, proba = precision_recall_curve(out_label_ids, preds[:, -1])
#         optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
#         print("Optimum threshold is:", optimal_proba_cutoff)
#         preds = [1 if i >= optimal_proba_cutoff else 0 for i in preds[:, -1]]
#     elif args['output_mode'] == "regression":
#         preds = np.squeeze(preds)
#     result, wrong = compute_metrics(EVAL_TASK, preds, out_label_ids)
#     results.update(result)

#     output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
#     with open(output_eval_file, "w") as writer:
#         logger.info("***** Eval results {} *****".format(prefix))
#         for key in sorted(result.keys()):
#             logger.info("  %s = %s", key, str(result[key]))
#             writer.write("%s = %s\n" % (key, str(result[key])))

#     return results, wrong

# args = json.loads('args.json')

# MODEL_CLASSES = {
#     'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
#     'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
#     'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
#     'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
# }

# config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]

# config = config_class.from_pretrained(args['model_name'], num_labels=2, finetuning_task=args['task_name'])
# tokenizer = tokenizer_class.from_pretrained(args['model_name'])

# model = model_class.from_pretrained(args['model_name'])
# model.to(device);

# task = args['task_name']

# processor = processors[task]()
# label_list = processor.get_labels()
# num_labels = len(label_list)

# ### Optional ###

# if args['do_train']:
#     train_dataset = load_and_cache_examples(task, tokenizer, undersample_scale_factor=0.1)
#     global_step, tr_loss = train(train_dataset, model, tokenizer)
#     logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

# ### Optional ###

# if args['do_train']:
#     if not os.path.exists(args['output_dir']):
#             os.makedirs(args['output_dir'])
#     logger.info("Saving model checkpoint to %s", args['output_dir'])
    
#     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
#     model_to_save.save_pretrained(args['output_dir'])
#     tokenizer.save_pretrained(args['output_dir'])
#     torch.save(args, os.path.join(args['output_dir'], 'training_args.bin'))

# results = {}
# if args['do_eval']:
#     checkpoints = [args['output_dir']]
#     if args['eval_all_checkpoints']:
#         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
#         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
#     logger.info("Evaluate the following checkpoints: %s", checkpoints)
#     for checkpoint in checkpoints:
#         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
#         model = model_class.from_pretrained(checkpoint)
#         model.to(device)
#         result, wrong_preds = evaluate(model, tokenizer, prefix=global_step)
#         result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
#         results.update(result)

# print(results)