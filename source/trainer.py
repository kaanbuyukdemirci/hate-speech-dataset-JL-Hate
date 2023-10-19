import torch
import transformers
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import shutil

try:
    from read_data import read_dataset
    from commons import *
except:
    from source.read_data import read_dataset
    from source.commons import *

# clear checkpoints
shutil.rmtree(CHECKPOINT)

# read, shuffle, and split dataset
turkish_dataset, english_dataset = read_dataset()

turkish_dataset.shuffle()
english_dataset.shuffle()

print("Datasets are initialized.")

# data collators
turkish_data_collator = transformers.DefaultDataCollator(return_tensors='pt')
english_data_collator = transformers.DefaultDataCollator(return_tensors='pt')
print("Data collators initialized.")

# trainer with cross validation
k_fold = 10
for k in range(0, k_fold):
    print(f"Training {k+1}. fold...")
    # model
    transformers.logging.set_verbosity_error()
    turkish_model = JointModel('tr')
    english_model = JointModel('en')
    print("Models initialized.")
    
    # training arguments, make it so that it saves the model at the end of the training
    turkish_training_args = transformers.TrainingArguments(output_dir=TURKISH_CHECK_POINT, overwrite_output_dir=True, 
                                                           evaluation_strategy='epoch', optim='adamw_torch', num_train_epochs=20,
                                                           save_strategy='no', save_steps=10000000000, save_total_limit=1,
                                                           log_level='critical')
    english_training_args = transformers.TrainingArguments(output_dir=ENGLISH_CHECK_POINT, overwrite_output_dir=True, 
                                                           evaluation_strategy='epoch', optim='adamw_torch', num_train_epochs=20,
                                                           save_strategy='no', save_steps=10000000000, save_total_limit=1,
                                                           log_level='critical')
    print("Training arguments initialized.")
    turkish_dataset_train, turkish_dataset_test = turkish_dataset.split(split_ratio=1-1/k_fold, k=k)
    english_dataset_train, english_dataset_test = english_dataset.split(split_ratio=1-1/k_fold, k=k)
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
    turkish_trainer.train()
    english_trainer.train()

    # save pytorch models
    #torch.save(turkish_model.state_dict(), TURKISH_CHECK_POINT + 'model.pt')
    #torch.save(english_model.state_dict(), ENGLISH_CHECK_POINT + 'model.pt')

    # plot loss history with respect to steps
    turkish_eval_loss = [x['eval_loss'] for x in turkish_trainer.state.log_history if 'eval_loss' in x.keys()]
    turkish_train_loss = [x['loss'] for x in turkish_trainer.state.log_history if 'loss' in x.keys()]
    english_eval_loss = [x['eval_loss'] for x in english_trainer.state.log_history if 'eval_loss' in x.keys()]
    english_train_loss = [x['loss'] for x in english_trainer.state.log_history if 'loss' in x.keys()]
    plt.figure(k, figsize=(10, 5))
    plt.plot(turkish_train_loss, label='Turkish Train')
    plt.plot(english_train_loss, label='English Train')
    plt.plot(turkish_eval_loss, label='Turkish Eval')
    plt.plot(english_eval_loss, label='English Eval')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{CHECKPOINT}/loss_history_{k}.png')


