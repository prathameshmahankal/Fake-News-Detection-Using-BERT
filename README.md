# Fake-News-Detection-Using-BERT

## Overview

This project tries to classify tweets as real or fake depending on the tweet text and also the text present in the article that is tagged in the tweet. This particular implementation uses BERT for classification.

Porject Website: https://prathameshmahankal.github.io/tracking-online-disinformation/

## Build Steps

Follow these steps to build this project in your local system:
1. Install all the dependencies using the **requirements.txt** file
2. The **data_preparation** notebook is used to create the final training dataset. You can directly extact the **data zip file** and find the **training dataset** there. You can also upload this folder to Azure using the **upload_data** script
3. Next, you could add any more data to the existing dataset and can use this combined data to train your BERT model in your local using the **train notebook**. This noteboook is a direct adaptation of this notebook built by Prateek Joshi (https://github.com/prateekjoshi565/Fine-Tuning-BERT).
4. Conversely, you can also train the model on Azure using the **run-training-script** file which internally runs the **train** script in the src folder.
5. Once the model is ready you can register the model in Azure and can then build you **echo_score** script where you write the code to initialize your model and generate predictions.
6. This script will be used while deploying your model using the **deploy** script. Note that the existing code currently deploys code on **Azure Container Instances**, but you can also deploy it on Azure Kubernetes Services with minimal code change.
7. Once all these steps are done, you have an ACI endpoint ready which you can then use to generate predictions directly for your test data.
8. Next, you need to build your **inference pipeline**. In this case, the **predict** script is the one that takes in data from blob storage and generates predictions for that file. In case you have a dataset that you wish to generate predictions for, you could use the **inference** script instead.
9. I also have a **receiver** script which waits for a message from the data extraction pipeline (handled by a different team member) and runs the predict script whenever data for any request is ready in the blob storage. This script might not be useful for someone not related to the project.

## Datasets Used
* LIAR
* ISOT Fake News
* Buzzfeed Political News Data
* Russian Troll tweets
* Fake news dataset by UTK Machine Learning Club
* Kaggle Fake News Detection
* NBC Election Troll Tweets 2016
* Fake News on Twitter 2016 viral tweets

## Demo Video

link

## Steps to run a demo

1. 

