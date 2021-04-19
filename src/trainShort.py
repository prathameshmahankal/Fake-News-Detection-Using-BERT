import pickle
import os
import time
import random
import argparse
import pandas as pd
import numpy as np
import joblib
import itertools
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt

from azureml.core import Workspace, Run, Dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_path',
    type=str,
    help='Path to the training data'
)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-5,
    help='Learning rate for SGD'
)
parser.add_argument(
    '--batch_size', 
    type=int, 
    dest='batch_size', 
    default=32
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
    default=5)

args = parser.parse_args()

batch_size = args.batch_size
learning_rate = args.learning_rate
adam_epsilon = args.adam_epsilon
num_epochs = args.num_epochs

run = Run.get_context()

print("===== DATA =====")
print("DATA PATH: " + args.data_path)
print("LIST FILES IN DATA PATH...")
print(os.listdir(args.data_path))
print("================")# ws = run.experiment.workspace

# # get the input dataset by ID
# dataset = Dataset.get_by_id(ws, id=args.data_path)

# # load the TabularDataset to pandas DataFrame
# df = dataset.to_pandas_dataframe()

df = pd.read_csv(args.data_path+'/shorttextpreprocessedtrain.csv')

counts = df.label.value_counts()
n = int(counts[counts==min(counts)]*0.8)
df = pd.concat([df[df.label==1].sample(n), df[df.label==0].sample(n)])
print(df.shape)

label_counts = pd.DataFrame(df['label'].value_counts())

label_values = list(label_counts.index)
order = list(pd.DataFrame(df['label'].value_counts()).index)
label_values = [l for _,l in sorted(zip(order, label_values))]

texts = df['text'].values
labels = df['label'].values

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

text_ids = [tokenizer.encode(text, max_length=40, pad_to_max_length=True) for text in texts]

att_masks = []
for ids in text_ids:
    masks = [int(id > 0) for id in ids]
    att_masks.append(masks)

# train_x, test_val_x, train_y, test_val_y = train_test_split(text_ids, labels, random_state=111, test_size=0.2)
# train_m, test_val_m = train_test_split(att_masks, random_state=111, test_size=0.2)
# test_x, val_x, test_y, val_y = train_test_split(test_val_x, test_val_y, random_state=111, test_size=0.5)
# test_m, val_m = train_test_split(test_val_m, random_state=111, test_size=0.5)

combined = [[x,y] for x,y in zip(text_ids,att_masks)]

train_x, test_val_x, train_y, test_val_y = train_test_split(combined, labels, random_state=111, test_size=0.2, stratify=labels)
test_x, val_x, test_y, val_y = train_test_split(test_val_x, test_val_y, random_state=111, test_size=0.5, stratify=test_val_y)

train_m = [element[1] for element in train_x]
train_x = [element[0] for element in train_x]

val_m = [element[1] for element in val_x]
val_x = [element[0] for element in val_x]

test_m = [element[1] for element in test_x]
test_x = [element[0] for element in test_x]

train_x = torch.tensor(train_x)
test_x = torch.tensor(test_x)
val_x = torch.tensor(val_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)
val_y = torch.tensor(val_y)
train_m = torch.tensor(train_m)
test_m = torch.tensor(test_m)
val_m = torch.tensor(val_m)

batch_size = 32

train_data = TensorDataset(train_x, train_m, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_x, val_m, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

num_labels = len(set(labels))

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
                                                            output_attentions=False, output_hidden_states=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.2},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

total_steps = len(train_dataloader) * num_epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

seed_val = 111
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

train_losses = []
val_losses = []
num_mb_train = len(train_dataloader)
num_mb_val = len(val_dataloader)

if num_mb_val == 0:
    num_mb_val = 1

for n in range(num_epochs):
    train_loss = 0
    val_loss = 0
    start_time = time.time()
    
    for k, (mb_x, mb_m, mb_y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        model.train()
        
        mb_x = mb_x.to(device)
        mb_m = mb_m.to(device)
        mb_y = mb_y.to(device)
        
        outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)
        
        loss = outputs[0]
        #loss = model_loss(outputs[1], mb_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.data / num_mb_train
    
    print ("\nTrain loss after iteration %i: %f" % (n+1, train_loss))
    train_losses.append(train_loss.cpu())
    
    with torch.no_grad():
        model.eval()
        
        for k, (mb_x, mb_m, mb_y) in enumerate(val_dataloader):
            mb_x = mb_x.to(device)
            mb_m = mb_m.to(device)
            mb_y = mb_y.to(device)
        
            outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)
            
            loss = outputs[0]
            #loss = model_loss(outputs[1], mb_y)
            
            val_loss += loss.data / num_mb_val
            
        print ("Validation loss after iteration %i: %f" % (n+1, val_loss))
        val_losses.append(val_loss.cpu())
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Time: {epoch_mins}m {epoch_secs}s')

    out_dir = './outputs'

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    
    with open(out_dir + '/train_losses.pkl', 'wb') as f:
        joblib.dump(train_losses, f)
    with open(out_dir + '/val_losses.pkl', 'wb') as f:
        joblib.dump(val_losses, f)

    # try:
    #     run.log('validation loss', val_losses)
    # except:
    #     pass

out_dir = './outputs'

# model = DistilBertForSequenceClassification.from_pretrained(out_dir)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

with open(out_dir + '/train_losses.pkl', 'rb') as f:
    train_losses = pickle.load(f)
    
with open(out_dir + '/val_losses.pkl', 'rb') as f:
    val_losses = pickle.load(f)    

plt.plot(train_losses, label="training loss")
plt.plot(val_losses, label="validation loss")
plt.legend()
plt.savefig(out_dir + '/losses.png')


test_data = TensorDataset(test_x, test_m)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

outputs = []
with torch.no_grad():
    model.eval()
    for k, (mb_x, mb_m) in enumerate(test_dataloader):
        mb_x = mb_x.to(device)
        mb_m = mb_m.to(device)
        output = model(mb_x, attention_mask=mb_m)
        outputs.append(output[0].to('cpu'))

outputs = torch.cat(outputs)

_, predicted_values = torch.max(outputs, 1)
predicted_values = predicted_values.numpy()
true_values = test_y.numpy()

test_accuracy = np.sum(predicted_values == true_values) / len(true_values)
print ("Test Accuracy:", test_accuracy)

# plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_dir + '/confusion-matrix.png')

cm_test = confusion_matrix(true_values, predicted_values)

np.set_printoptions(precision=2)

plt.figure(figsize=(6,6))
plot_confusion_matrix(cm_test, classes=label_values, title='Confusion Matrix - Test Dataset', normalize=True)

# kwargs = {'num_workers': 1, 'pin_memory': True} if gpu_available else {}

# hvd.init()


# # prepare DataLoader for CIFAR10 data
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# trainset = torchvision.datasets.CIFAR10(
#     root=args.data_path,
#     train=True,
#     download=False,
#     transform=transform,
# )
# trainloader = torch.utils.data.DataLoader(
#     trainset,
#     batch_size=4,
#     shuffle=True,
#     num_workers=2
# )

# # define convolutional network
# net = Net()

# # set up pytorch loss /  optimizer
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(
#     net.parameters(),
#     lr=args.learning_rate,
#     momentum=args.momentum,
# )

# # train the network
# for epoch in range(2):

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # unpack the data
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:
#             loss = running_loss / 2000
#             run.log('loss', loss)  # log loss metric to AML
#             print(f'epoch={epoch + 1}, batch={i + 1:5}: loss {loss:.2f}')
#             running_loss = 0.0

# print('Finished Training')