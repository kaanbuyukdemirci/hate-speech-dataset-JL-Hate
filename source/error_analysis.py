import torch
import transformers
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import shutil
from typing import Literal

try:
    from read_data import read_dataset
    from commons import *
except:
    from source.read_data import read_dataset
    from source.commons import *

# language
lng:Literal['tr', 'en', 'both'] = 'tr'

# read, shuffle, and split dataset
turkish_dataset, english_dataset = read_dataset()

if lng == 'tr':
    turkish_dataset.shuffle()
    del english_dataset
elif lng == 'en':
    english_dataset.shuffle()
    del turkish_dataset
elif lng == 'both':
    turkish_dataset.shuffle()
    english_dataset.shuffle()

print("Datasets are initialized.")

# data collators
if lng == 'tr':
    turkish_data_collator = transformers.DefaultDataCollator(return_tensors='pt')
elif lng == 'en':
    english_data_collator = transformers.DefaultDataCollator(return_tensors='pt')
elif lng == 'both':
    turkish_data_collator = transformers.DefaultDataCollator(return_tensors='pt')
    english_data_collator = transformers.DefaultDataCollator(return_tensors='pt')
print("Data collators initialized.")

# trainer with cross validation
k_fold = 10
k = 0

# model
transformers.logging.set_verbosity_error()
if lng == 'tr':
    turkish_model = JointModel('tr')
elif lng == 'en':
    english_model = JointModel('en')
elif lng == 'both':
    turkish_model = JointModel('tr')
    english_model = JointModel('en')
print("Models initialized.")

# training arguments, make it so that it saves the model at the end of the training
if lng == 'tr':
    turkish_training_args = transformers.TrainingArguments(output_dir=TURKISH_CHECK_POINT, overwrite_output_dir=True, 
                                                        evaluation_strategy='epoch', save_strategy='epoch',
                                                        optim='adamw_torch', num_train_epochs=1,
                                                        save_total_limit=1, log_level='critical', auto_find_batch_size=True,
                                                        load_best_model_at_end=True, metric_for_best_model='eval_f1_score_token_macro',
                                                        greater_is_better=True)
elif lng == 'en':
    english_training_args = transformers.TrainingArguments(output_dir=TURKISH_CHECK_POINT, overwrite_output_dir=True, 
                                                        evaluation_strategy='epoch', save_strategy='epoch',
                                                        optim='adamw_torch', num_train_epochs=1,
                                                        save_total_limit=1, log_level='critical', auto_find_batch_size=True,
                                                        load_best_model_at_end=True, metric_for_best_model='eval_f1_score_token_macro',
                                                        greater_is_better=True)
elif lng == 'both':
    turkish_training_args = transformers.TrainingArguments(output_dir=TURKISH_CHECK_POINT, overwrite_output_dir=True, 
                                                            evaluation_strategy='epoch', save_strategy='epoch',
                                                            optim='adamw_torch', num_train_epochs=1,
                                                            save_total_limit=1, log_level='critical', auto_find_batch_size=True,
                                                            load_best_model_at_end=True, metric_for_best_model='eval_f1_score_token_macro',
                                                            greater_is_better=True)
    english_training_args = transformers.TrainingArguments(output_dir=TURKISH_CHECK_POINT, overwrite_output_dir=True, 
                                                            evaluation_strategy='epoch', save_strategy='epoch',
                                                            optim='adamw_torch', num_train_epochs=1,
                                                            save_total_limit=1, log_level='critical', auto_find_batch_size=True,
                                                            load_best_model_at_end=True, metric_for_best_model='eval_f1_score_token_macro',
                                                            greater_is_better=True)
print("Training arguments initialized.")

# split dataset
if lng == 'tr':
    turkish_dataset_train, turkish_dataset_test = turkish_dataset.split(split_ratio=1-1/k_fold, k=k)
elif lng == 'en':
    english_dataset_train, english_dataset_test = english_dataset.split(split_ratio=1-1/k_fold, k=k)
elif lng == 'both':
    turkish_dataset_train, turkish_dataset_test = turkish_dataset.split(split_ratio=1-1/k_fold, k=k)
    english_dataset_train, english_dataset_test = english_dataset.split(split_ratio=1-1/k_fold, k=k)

# trainers
if lng == 'tr':
    turkish_trainer = transformers.Trainer(model=turkish_model, 
                                        args=turkish_training_args, 
                                        train_dataset=turkish_dataset_train, 
                                        eval_dataset=turkish_dataset_test, 
                                        data_collator=turkish_data_collator,
                                        compute_metrics=compute_metrics)
elif lng == 'en':
    english_trainer = transformers.Trainer(model=english_model, 
                                        args=english_training_args,
                                        train_dataset=english_dataset_train,
                                        eval_dataset=english_dataset_test, 
                                        data_collator=english_data_collator,
                                        compute_metrics=compute_metrics)
elif lng == 'both':
    turkish_trainer = transformers.Trainer(model=turkish_model, 
                                            args=turkish_training_args, 
                                            train_dataset=turkish_dataset_train, 
                                            eval_dataset=turkish_dataset_test, 
                                            data_collator=turkish_data_collator,
                                            compute_metrics=compute_metrics)
    english_trainer = transformers.Trainer(model=english_model, 
                                            args=english_training_args,
                                            train_dataset=english_dataset_train,
                                            eval_dataset=english_dataset_test, 
                                            data_collator=english_data_collator,
                                            compute_metrics=compute_metrics)
print("Trainers initialized.")

# train
if lng == 'tr':
    turkish_trainer.train()
elif lng == 'en':
    english_trainer.train()
elif lng == 'both':
    turkish_trainer.train()
    english_trainer.train()

# find the errors
new_device = turkish_model.sequence_classifier.weight.device
if lng == 'tr':
    turkish_dataset_test = turkish_dataset_test.to(new_device)
    turkish_predicted_labels = turkish_model.predict(turkish_dataset_test[:]['input_ids'], turkish_dataset_test[:]['attention_mask'])
    turkish_predicted_sequence_labels = turkish_predicted_labels['sequence_predictions']
    turkish_predicted_token_labels = turkish_predicted_labels['token_predictions']
    turkish_actual_sequence_labels = turkish_dataset_test[:]['sequence_label']
    turkish_actual_token_labels = turkish_dataset_test[:]['token_label']
    turkish_sequence_errors = []
    turkish_token_errors = []
    for i in range(len(turkish_actual_sequence_labels)):
        if turkish_actual_sequence_labels[i] != turkish_predicted_sequence_labels[i]:
            turkish_sequence_errors.append(i)
        else:
            turkish_token_errors.append(i)
elif lng == 'en':
    english_dataset_test = english_dataset_test.to(new_device)
    english_predicted_labels = english_model.predict(english_dataset_test[:]['input_ids'], english_dataset_test[:]['attention_mask'])
    english_predicted_sequence_labels = english_predicted_labels['sequence_predictions']
    english_predicted_token_labels = english_predicted_labels['token_predictions']
    english_actual_sequence_labels = english_dataset_test[:]['sequence_label']
    english_actual_token_labels = english_dataset_test[:]['token_label']
    english_sequence_errors = []
    english_token_errors = []
    for i in range(len(english_actual_sequence_labels)):
        if english_actual_sequence_labels[i] != english_predicted_sequence_labels[i]:
            english_sequence_errors.append(i)
        else:
            english_token_errors.append(i)
elif lng == 'both':
    turkish_dataset_test = turkish_dataset_test.to(new_device)
    turkish_predicted_labels = turkish_model.predict(turkish_dataset_test[:]['input_ids'], turkish_dataset_test[:]['attention_mask'])
    turkish_predicted_sequence_labels = turkish_predicted_labels['sequence_predictions']
    turkish_predicted_token_labels = turkish_predicted_labels['token_predictions']
    turkish_actual_sequence_labels = turkish_dataset_test[:]['sequence_label']
    turkish_actual_token_labels = turkish_dataset_test[:]['token_label']
    turkish_sequence_errors = []
    turkish_token_errors = []
    for i in range(len(turkish_actual_sequence_labels)):
        if turkish_actual_sequence_labels[i] != turkish_predicted_sequence_labels[i]:
            turkish_sequence_errors.append(i)
        else:
            turkish_token_errors.append(i)
    
    english_dataset_test = english_dataset_test.to(new_device)
    english_predicted_labels = english_model.predict(english_dataset_test[:]['input_ids'], english_dataset_test[:]['attention_mask'])
    english_predicted_sequence_labels = english_predicted_labels['sequence_predictions']
    english_predicted_token_labels = english_predicted_labels['token_predictions']
    english_actual_sequence_labels = english_dataset_test[:]['sequence_label']
    english_actual_token_labels = english_dataset_test[:]['token_label']
    english_sequence_errors = []
    english_token_errors = []
    for i in range(len(english_actual_sequence_labels)):
        if english_actual_sequence_labels[i] != english_predicted_sequence_labels[i]:
            english_sequence_errors.append(i)
        else:
            english_token_errors.append(i)

# print 1 sequence 1 token errors
if lng == 'tr':
    i = 0
    while True:
        print("Turkish sequence errors: (text, actual, predicted)")
        print(turkish_dataset_test[turkish_sequence_errors[i]]['text'])
        print(turkish_dataset_test[turkish_sequence_errors[i]]['sequence_label'])
        print(turkish_predicted_sequence_labels[turkish_sequence_errors[i]])
        print("Turkish token errors:")
        print("tokenized text - actual token labels - predicted token labels")
        for j in range(len(turkish_dataset_test[turkish_token_errors[i]]['text'])):
            print(turkish_dataset_test[turkish_token_errors[i]]['text'][j], turkish_dataset_test[turkish_token_errors[i]]['token_label'][j], turkish_predicted_token_labels[turkish_token_errors[i]][j])
        print(turkish_dataset_test[turkish_token_errors[i]]['text'])
        print(turkish_dataset_test[turkish_token_errors[i]]['token_label'])
        print(turkish_predicted_token_labels[turkish_token_errors[i]])
        print("Press enter to continue...")
        input()
        print("----------------------------------------")
elif lng == 'en':
    i = 0
    while True:
        print("English sequence errors:")
        print(english_dataset_test[english_sequence_errors[i]]['text'])
        print(english_dataset_test[english_sequence_errors[i]]['sequence_label'])
        print(english_predicted_sequence_labels[english_sequence_errors[i]])
        print("English token errors:")
        print("tokenized text - actual token labels - predicted token labels")
        for j in range(len(english_dataset_test[english_token_errors[i]]['text'])):
            print(english_dataset_test[english_token_errors[i]]['text'][j], english_dataset_test[english_token_errors[i]]['token_label'][j], english_predicted_token_labels[english_token_errors[i]][j])
        print(english_dataset_test[english_token_errors[i]]['text'])
        print(english_dataset_test[english_token_errors[i]]['token_label'])
        print(english_predicted_token_labels[english_token_errors[i]])
        print("Press enter to continue...")
        input()
        print("----------------------------------------")
elif lng == 'both':
    i = 0
    while True:
        print("Turkish sequence errors: (text, actual, predicted)")
        print(turkish_dataset_test[turkish_sequence_errors[i]]['text'])
        print(turkish_dataset_test[turkish_sequence_errors[i]]['sequence_label'])
        print(turkish_predicted_sequence_labels[turkish_sequence_errors[i]])
        print("Turkish token errors:")
        print("tokenized text - actual token labels - predicted token labels")
        for j in range(len(turkish_dataset_test[turkish_token_errors[i]]['text'])):
            print(turkish_dataset_test[turkish_token_errors[i]]['text'][j], turkish_dataset_test[turkish_token_errors[i]]['token_label'][j], turkish_predicted_token_labels[turkish_token_errors[i]][j])
        print(turkish_dataset_test[turkish_token_errors[i]]['text'])
        print(turkish_dataset_test[turkish_token_errors[i]]['token_label'])
        print(turkish_predicted_token_labels[turkish_token_errors[i]])
        print("English sequence errors:")
        print(english_dataset_test[english_sequence_errors[i]]['text'])
        print(english_dataset_test[english_sequence_errors[i]]['sequence_label'])
        print(english_predicted_sequence_labels[english_sequence_errors[i]])
        print("English token errors:")
        print("tokenized text - actual token labels - predicted token labels")
        for j in range(len(english_dataset_test[english_token_errors[i]]['text'])):
            print(english_dataset_test[english_token_errors[i]]['text'][j], english_dataset_test[english_token_errors[i]]['token_label'][j], english_predicted_token_labels[english_token_errors[i]][j])
        print(english_dataset_test[english_token_errors[i]]['text'])
        print(english_dataset_test[english_token_errors[i]]['token_label'])
        print(english_predicted_token_labels[english_token_errors[i]])
        print("Press enter to continue...")
        input()
        print("----------------------------------------")



