import os
import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
import json
# import torch.optim as optim
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertConfig
# from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, f1_score, recall_score, precision_score, 
                             precision_recall_curve, confusion_matrix, matthews_corrcoef)
import matplotlib.pyplot as plt
from azureml.core import Workspace, Run, Dataset
from azureml.core.model import Model
import warnings
warnings.filterwarnings("ignore")

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

def init():

	print('This is init')
	
	global model
	global device

	# paramters
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# import BERT-base pretrained model
	bert = AutoModel.from_pretrained('bert-base-uncased')

	# pass the pre-trained BERT to our define architecture
	model = BERT_Arch(bert)
	model = model.to(device)

	print(os.getenv('AZUREML_MODEL_DIR'))
	print(os.path.join(os.getenv('AZUREML_MODEL_DIR'),'saved_weights.pt'))

	#load weights of best model
	path = '../outputs/saved_weights.pt'
	model.load_state_dict(torch.load(os.path.join(os.getenv('AZUREML_MODEL_DIR'),'saved_weights.pt')))

def run(request):

	test = json.loads(request)
	# print(f'received data {test}')

	df = pd.DataFrame(test)
	df.dropna(inplace=True)
	# print(df.head())

	test_text = df.text
	# test_labels = df.label

	# Load the BERT tokenizer
	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

	max_seq_len = 25

	# tokenize and encode sequences in the test set
	tokens_test = tokenizer.batch_encode_plus(
	    test_text.tolist(),
	    max_length = max_seq_len,
	    pad_to_max_length=True,
	    truncation=True,
	    return_token_type_ids=False
	)

	# for test set
	test_seq = torch.tensor(tokens_test['input_ids'])
	test_mask = torch.tensor(tokens_test['attention_mask'])
	# test_y = torch.tensor(test_labels.tolist())

	# get predictions for test data
	with torch.no_grad():
	  preds = model(test_seq.to(device), test_mask.to(device))
	  preds = preds.detach().cpu().numpy()

	# model's performance
	# precision_, recall_, proba = precision_recall_curve(test_y, preds[:, -1])
	preds = np.argmax(preds, axis = 1)

	# #plot precision-recall curve
	# plt.plot(recall_, precision_, marker='.', label='BERT-model')
	# plt.xlabel('Recall')
	# plt.ylabel('Precision')
	# plt.legend()
	# plt.show()

	# optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
	# preds = [1 if i >= optimal_proba_cutoff else 0 for i in preds[:, -1]]

	# mcc = matthews_corrcoef(test_y, preds)
	# tn, fp, fn, tp = confusion_matrix(test_y, preds).ravel()
	# precision = precision_score(test_y, preds)
	# recall = recall_score(test_y, preds)
	# f1 = f1_score(test_y, preds, average='weighted')

	# print("")
	# print("Matthews Corr Coef:", mcc)
	# print("Precision:", precision)
	# print("Recall:", recall)
	# print("f-1 score:", f1)

	# print(classification_report(test_y, preds))

	return preds.tolist()

# if __name__=='__main__':
# 	init()

# 	df = pd.read_csv("../data/newdatasetwithcoviddata.csv")
# 	df = df.sample(500)
	
# 	run(df.to_json())