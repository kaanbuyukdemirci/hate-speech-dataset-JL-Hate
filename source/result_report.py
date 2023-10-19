import torch
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from commons import *
import os

# extract all event files found in log_dir
turkish_log_dir = TURKISH_CHECK_POINT + f'/runs'
english_log_dir = ENGLISH_CHECK_POINT + f'/runs'

turkish_event_files = []
for x in os.listdir(turkish_log_dir):
    event_loc = os.listdir(turkish_log_dir + '/' + x)[0]
    turkish_event_files.append(turkish_log_dir + '/' + x + '/' + event_loc)
english_event_files = []
for x in os.listdir(english_log_dir):
    event_loc = os.listdir(english_log_dir + '/' + x)[0]
    english_event_files.append(english_log_dir + '/' + x + '/' + event_loc)

print(turkish_event_files)
print(english_event_files)

# extract eval/f1_score_sequence_macro, eval/f1_score_token_macro
# eval/f1_score_sequence_0, eval/f1_score_sequence_1, eval/f1_score_sequence_2
# eval/f1_score_token_0, eval/f1_score_token_1, eval/f1_score_token_2, eval/f1_score_token_3
# eval/f1_score_token_4, eval/f1_score_token_5, eval/f1_score_token_6, eval/f1_score_token_7
# eval/f1_score_token_8
turkish_all_f1_score_sequence_macro = []
turkish_all_f1_score_token_macro = []
turkish_all_f1_score_sequence_0 = []
turkish_all_f1_score_sequence_1 = []
turkish_all_f1_score_sequence_2 = []
turkish_all_f1_score_token_0 = []
turkish_all_f1_score_token_1 = []
turkish_all_f1_score_token_2 = []
turkish_all_f1_score_token_3 = []
turkish_all_f1_score_token_4 = []
if TAG_TYPE == 'BIO':
    turkish_all_f1_score_token_5 = []
    turkish_all_f1_score_token_6 = []
    turkish_all_f1_score_token_7 = []
    turkish_all_f1_score_token_8 = []

english_all_f1_score_sequence_macro = []
english_all_f1_score_token_macro = []
english_all_f1_score_sequence_0 = []
english_all_f1_score_sequence_1 = []
english_all_f1_score_sequence_2 = []
english_all_f1_score_token_0 = []
english_all_f1_score_token_1 = []
english_all_f1_score_token_2 = []
english_all_f1_score_token_3 = []
english_all_f1_score_token_4 = []
if TAG_TYPE == 'BIO':
    english_all_f1_score_token_5 = []
    english_all_f1_score_token_6 = []
    english_all_f1_score_token_7 = []
    english_all_f1_score_token_8 = []

for x in turkish_event_files:
    event_acc = EventAccumulator(x)
    event_acc.Reload()
    turkish_all_f1_score_sequence_macro.append([x.value for x in event_acc.Scalars("eval/f1_score_sequence_macro")])
    turkish_all_f1_score_sequence_0.append([x.value for x in event_acc.Scalars("eval/f1_score_sequence_0")])
    turkish_all_f1_score_sequence_1.append([x.value for x in event_acc.Scalars("eval/f1_score_sequence_1")])
    turkish_all_f1_score_sequence_2.append([x.value for x in event_acc.Scalars("eval/f1_score_sequence_2")])
    turkish_all_f1_score_token_macro.append([x.value for x in event_acc.Scalars("eval/f1_score_token_macro")])
    turkish_all_f1_score_token_0.append([x.value for x in event_acc.Scalars("eval/f1_score_token_0")])
    turkish_all_f1_score_token_1.append([x.value for x in event_acc.Scalars("eval/f1_score_token_1")])
    turkish_all_f1_score_token_2.append([x.value for x in event_acc.Scalars("eval/f1_score_token_2")])
    turkish_all_f1_score_token_3.append([x.value for x in event_acc.Scalars("eval/f1_score_token_3")])
    turkish_all_f1_score_token_4.append([x.value for x in event_acc.Scalars("eval/f1_score_token_4")])
    if TAG_TYPE == 'BIO':
        turkish_all_f1_score_token_5.append([x.value for x in event_acc.Scalars("eval/f1_score_token_5")])
        turkish_all_f1_score_token_6.append([x.value for x in event_acc.Scalars("eval/f1_score_token_6")])
        turkish_all_f1_score_token_7.append([x.value for x in event_acc.Scalars("eval/f1_score_token_7")])
        turkish_all_f1_score_token_8.append([x.value for x in event_acc.Scalars("eval/f1_score_token_8")])

for x in english_event_files:
    event_acc = EventAccumulator(x)
    event_acc.Reload()
    english_all_f1_score_sequence_macro.append([x.value for x in event_acc.Scalars("eval/f1_score_sequence_macro")])
    english_all_f1_score_sequence_0.append([x.value for x in event_acc.Scalars("eval/f1_score_sequence_0")])
    english_all_f1_score_sequence_1.append([x.value for x in event_acc.Scalars("eval/f1_score_sequence_1")])
    english_all_f1_score_sequence_2.append([x.value for x in event_acc.Scalars("eval/f1_score_sequence_2")])
    english_all_f1_score_token_macro.append([x.value for x in event_acc.Scalars("eval/f1_score_token_macro")])
    english_all_f1_score_token_0.append([x.value for x in event_acc.Scalars("eval/f1_score_token_0")])
    english_all_f1_score_token_1.append([x.value for x in event_acc.Scalars("eval/f1_score_token_1")])
    english_all_f1_score_token_2.append([x.value for x in event_acc.Scalars("eval/f1_score_token_2")])
    english_all_f1_score_token_3.append([x.value for x in event_acc.Scalars("eval/f1_score_token_3")])
    english_all_f1_score_token_4.append([x.value for x in event_acc.Scalars("eval/f1_score_token_4")])
    if TAG_TYPE == 'BIO':
        english_all_f1_score_token_5.append([x.value for x in event_acc.Scalars("eval/f1_score_token_5")])
        english_all_f1_score_token_6.append([x.value for x in event_acc.Scalars("eval/f1_score_token_6")])
        english_all_f1_score_token_7.append([x.value for x in event_acc.Scalars("eval/f1_score_token_7")])
        english_all_f1_score_token_8.append([x.value for x in event_acc.Scalars("eval/f1_score_token_8")])

turkish_all_f1_score_sequence_macro = 100*np.array(turkish_all_f1_score_sequence_macro)
turkish_all_f1_score_sequence_0 = 100*np.array(turkish_all_f1_score_sequence_0)
turkish_all_f1_score_sequence_1 = 100*np.array(turkish_all_f1_score_sequence_1)
turkish_all_f1_score_sequence_2 = 100*np.array(turkish_all_f1_score_sequence_2)
turkish_all_f1_score_token_macro = 100*np.array(turkish_all_f1_score_token_macro)
turkish_all_f1_score_token_0 = 100*np.array(turkish_all_f1_score_token_0)
turkish_all_f1_score_token_1 = 100*np.array(turkish_all_f1_score_token_1)
turkish_all_f1_score_token_2 = 100*np.array(turkish_all_f1_score_token_2)
turkish_all_f1_score_token_3 = 100*np.array(turkish_all_f1_score_token_3)
turkish_all_f1_score_token_4 = 100*np.array(turkish_all_f1_score_token_4)
if TAG_TYPE == 'BIO':
    turkish_all_f1_score_token_5 = 100*np.array(turkish_all_f1_score_token_5)
    turkish_all_f1_score_token_6 = 100*np.array(turkish_all_f1_score_token_6)
    turkish_all_f1_score_token_7 = 100*np.array(turkish_all_f1_score_token_7)
    turkish_all_f1_score_token_8 = 100*np.array(turkish_all_f1_score_token_8)

english_all_f1_score_sequence_macro = 100*np.array(english_all_f1_score_sequence_macro)
english_all_f1_score_sequence_0 = 100*np.array(english_all_f1_score_sequence_0)
english_all_f1_score_sequence_1 = 100*np.array(english_all_f1_score_sequence_1)
english_all_f1_score_sequence_2 = 100*np.array(english_all_f1_score_sequence_2)
english_all_f1_score_token_macro = 100*np.array(english_all_f1_score_token_macro)
english_all_f1_score_token_0 = 100*np.array(english_all_f1_score_token_0)
english_all_f1_score_token_1 = 100*np.array(english_all_f1_score_token_1)
english_all_f1_score_token_2 = 100*np.array(english_all_f1_score_token_2)
english_all_f1_score_token_3 = 100*np.array(english_all_f1_score_token_3)
english_all_f1_score_token_4 = 100*np.array(english_all_f1_score_token_4)
if TAG_TYPE == 'BIO':
    english_all_f1_score_token_5 = 100*np.array(english_all_f1_score_token_5)
    english_all_f1_score_token_6 = 100*np.array(english_all_f1_score_token_6)
    english_all_f1_score_token_7 = 100*np.array(english_all_f1_score_token_7)
    english_all_f1_score_token_8 = 100*np.array(english_all_f1_score_token_8)

# take the max for each sample when token_macro reaches its max
result_type = RESULT_TYPE
if result_type == 'best':
    turkish_max_locations = np.argmax(turkish_all_f1_score_token_macro, axis=1)
    english_max_locations = np.argmax(english_all_f1_score_token_macro, axis=1)
else:
    turkish_max_locations = np.zeros(len(turkish_all_f1_score_token_macro), dtype=int) - 1
    english_max_locations = np.zeros(len(english_all_f1_score_token_macro), dtype=int) - 1
turkish_all_f1_score_sequence_macro = turkish_all_f1_score_sequence_macro[np.arange(len(turkish_all_f1_score_sequence_macro)), turkish_max_locations]
turkish_all_f1_score_sequence_0 = turkish_all_f1_score_sequence_0[np.arange(len(turkish_all_f1_score_sequence_0)), turkish_max_locations]
turkish_all_f1_score_sequence_1 = turkish_all_f1_score_sequence_1[np.arange(len(turkish_all_f1_score_sequence_1)), turkish_max_locations]
turkish_all_f1_score_sequence_2 = turkish_all_f1_score_sequence_2[np.arange(len(turkish_all_f1_score_sequence_2)), turkish_max_locations]
turkish_all_f1_score_token_macro = turkish_all_f1_score_token_macro[np.arange(len(turkish_all_f1_score_token_macro)), turkish_max_locations]
turkish_all_f1_score_token_0 = turkish_all_f1_score_token_0[np.arange(len(turkish_all_f1_score_token_0)), turkish_max_locations]
turkish_all_f1_score_token_1 = turkish_all_f1_score_token_1[np.arange(len(turkish_all_f1_score_token_1)), turkish_max_locations]
turkish_all_f1_score_token_2 = turkish_all_f1_score_token_2[np.arange(len(turkish_all_f1_score_token_2)), turkish_max_locations]
turkish_all_f1_score_token_3 = turkish_all_f1_score_token_3[np.arange(len(turkish_all_f1_score_token_3)), turkish_max_locations]
turkish_all_f1_score_token_4 = turkish_all_f1_score_token_4[np.arange(len(turkish_all_f1_score_token_4)), turkish_max_locations]
if TAG_TYPE == 'BIO':
    turkish_all_f1_score_token_5 = turkish_all_f1_score_token_5[np.arange(len(turkish_all_f1_score_token_5)), turkish_max_locations]
    turkish_all_f1_score_token_6 = turkish_all_f1_score_token_6[np.arange(len(turkish_all_f1_score_token_6)), turkish_max_locations]
    turkish_all_f1_score_token_7 = turkish_all_f1_score_token_7[np.arange(len(turkish_all_f1_score_token_7)), turkish_max_locations]
    turkish_all_f1_score_token_8 = turkish_all_f1_score_token_8[np.arange(len(turkish_all_f1_score_token_8)), turkish_max_locations]
    
english_all_f1_score_sequence_macro = english_all_f1_score_sequence_macro[np.arange(len(english_all_f1_score_sequence_macro)), english_max_locations]
english_all_f1_score_sequence_0 = english_all_f1_score_sequence_0[np.arange(len(english_all_f1_score_sequence_0)), english_max_locations]
english_all_f1_score_sequence_1 = english_all_f1_score_sequence_1[np.arange(len(english_all_f1_score_sequence_1)), english_max_locations]
english_all_f1_score_sequence_2 = english_all_f1_score_sequence_2[np.arange(len(english_all_f1_score_sequence_2)), english_max_locations]
english_all_f1_score_token_macro = english_all_f1_score_token_macro[np.arange(len(english_all_f1_score_token_macro)), english_max_locations]
english_all_f1_score_token_0 = english_all_f1_score_token_0[np.arange(len(english_all_f1_score_token_0)), english_max_locations]
english_all_f1_score_token_1 = english_all_f1_score_token_1[np.arange(len(english_all_f1_score_token_1)), english_max_locations]
english_all_f1_score_token_2 = english_all_f1_score_token_2[np.arange(len(english_all_f1_score_token_2)), english_max_locations]
english_all_f1_score_token_3 = english_all_f1_score_token_3[np.arange(len(english_all_f1_score_token_3)), english_max_locations]
english_all_f1_score_token_4 = english_all_f1_score_token_4[np.arange(len(english_all_f1_score_token_4)), english_max_locations]
if TAG_TYPE == 'BIO':
    english_all_f1_score_token_5 = english_all_f1_score_token_5[np.arange(len(english_all_f1_score_token_5)), english_max_locations]
    english_all_f1_score_token_6 = english_all_f1_score_token_6[np.arange(len(english_all_f1_score_token_6)), english_max_locations]
    english_all_f1_score_token_7 = english_all_f1_score_token_7[np.arange(len(english_all_f1_score_token_7)), english_max_locations]
    english_all_f1_score_token_8 = english_all_f1_score_token_8[np.arange(len(english_all_f1_score_token_8)), english_max_locations]

# find mean and std
turkish_all_f1_score_sequence_macro = np.mean(turkish_all_f1_score_sequence_macro), np.std(turkish_all_f1_score_sequence_macro)
turkish_all_f1_score_sequence_0 = np.mean(turkish_all_f1_score_sequence_0), np.std(turkish_all_f1_score_sequence_0)
turkish_all_f1_score_sequence_1 = np.mean(turkish_all_f1_score_sequence_1), np.std(turkish_all_f1_score_sequence_1)
turkish_all_f1_score_sequence_2 = np.mean(turkish_all_f1_score_sequence_2), np.std(turkish_all_f1_score_sequence_2)
turkish_all_f1_score_token_macro = np.mean(turkish_all_f1_score_token_macro), np.std(turkish_all_f1_score_token_macro)
turkish_all_f1_score_token_0 = np.mean(turkish_all_f1_score_token_0), np.std(turkish_all_f1_score_token_0)
turkish_all_f1_score_token_1 = np.mean(turkish_all_f1_score_token_1), np.std(turkish_all_f1_score_token_1)
turkish_all_f1_score_token_2 = np.mean(turkish_all_f1_score_token_2), np.std(turkish_all_f1_score_token_2)
turkish_all_f1_score_token_3 = np.mean(turkish_all_f1_score_token_3), np.std(turkish_all_f1_score_token_3)
turkish_all_f1_score_token_4 = np.mean(turkish_all_f1_score_token_4), np.std(turkish_all_f1_score_token_4)
if TAG_TYPE == 'BIO':
    turkish_all_f1_score_token_5 = np.mean(turkish_all_f1_score_token_5), np.std(turkish_all_f1_score_token_5)
    turkish_all_f1_score_token_6 = np.mean(turkish_all_f1_score_token_6), np.std(turkish_all_f1_score_token_6)
    turkish_all_f1_score_token_7 = np.mean(turkish_all_f1_score_token_7), np.std(turkish_all_f1_score_token_7)
    turkish_all_f1_score_token_8 = np.mean(turkish_all_f1_score_token_8), np.std(turkish_all_f1_score_token_8)

english_all_f1_score_sequence_macro = np.mean(english_all_f1_score_sequence_macro), np.std(english_all_f1_score_sequence_macro)
english_all_f1_score_sequence_0 = np.mean(english_all_f1_score_sequence_0), np.std(english_all_f1_score_sequence_0)
english_all_f1_score_sequence_1 = np.mean(english_all_f1_score_sequence_1), np.std(english_all_f1_score_sequence_1)
english_all_f1_score_sequence_2 = np.mean(english_all_f1_score_sequence_2), np.std(english_all_f1_score_sequence_2)
english_all_f1_score_token_macro = np.mean(english_all_f1_score_token_macro), np.std(english_all_f1_score_token_macro)
english_all_f1_score_token_0 = np.mean(english_all_f1_score_token_0), np.std(english_all_f1_score_token_0)
english_all_f1_score_token_1 = np.mean(english_all_f1_score_token_1), np.std(english_all_f1_score_token_1)
english_all_f1_score_token_2 = np.mean(english_all_f1_score_token_2), np.std(english_all_f1_score_token_2)
english_all_f1_score_token_3 = np.mean(english_all_f1_score_token_3), np.std(english_all_f1_score_token_3)
english_all_f1_score_token_4 = np.mean(english_all_f1_score_token_4), np.std(english_all_f1_score_token_4)
if TAG_TYPE == 'BIO':
    english_all_f1_score_token_5 = np.mean(english_all_f1_score_token_5), np.std(english_all_f1_score_token_5)
    english_all_f1_score_token_6 = np.mean(english_all_f1_score_token_6), np.std(english_all_f1_score_token_6)
    english_all_f1_score_token_7 = np.mean(english_all_f1_score_token_7), np.std(english_all_f1_score_token_7)
    english_all_f1_score_token_8 = np.mean(english_all_f1_score_token_8), np.std(english_all_f1_score_token_8)

# round and print
round_to = 1
multiply_by = 1
print('turkish_all_f1_score_sequence_macro', round(turkish_all_f1_score_sequence_macro[0], round_to), round(turkish_all_f1_score_sequence_macro[1], round_to))
print('turkish_all_f1_score_sequence_0', round(turkish_all_f1_score_sequence_0[0], round_to), round(turkish_all_f1_score_sequence_0[1], round_to))
print('turkish_all_f1_score_sequence_1', round(turkish_all_f1_score_sequence_1[0], round_to), round(turkish_all_f1_score_sequence_1[1], round_to))
print('turkish_all_f1_score_sequence_2', round(turkish_all_f1_score_sequence_2[0], round_to), round(turkish_all_f1_score_sequence_2[1], round_to))
print('turkish_all_f1_score_token_macro', round(turkish_all_f1_score_token_macro[0], round_to), round(turkish_all_f1_score_token_macro[1], round_to))
print('turkish_all_f1_score_token_0', round(turkish_all_f1_score_token_0[0], round_to), round(turkish_all_f1_score_token_0[1], round_to))
print('turkish_all_f1_score_token_1', round(turkish_all_f1_score_token_1[0], round_to), round(turkish_all_f1_score_token_1[1], round_to))
print('turkish_all_f1_score_token_2', round(turkish_all_f1_score_token_2[0], round_to), round(turkish_all_f1_score_token_2[1], round_to))
print('turkish_all_f1_score_token_3', round(turkish_all_f1_score_token_3[0], round_to), round(turkish_all_f1_score_token_3[1], round_to))
print('turkish_all_f1_score_token_4', round(turkish_all_f1_score_token_4[0], round_to), round(turkish_all_f1_score_token_4[1], round_to))
if TAG_TYPE == 'BIO':
    print('turkish_all_f1_score_token_5', round(turkish_all_f1_score_token_5[0], round_to), round(turkish_all_f1_score_token_5[1], round_to))
    print('turkish_all_f1_score_token_6', round(turkish_all_f1_score_token_6[0], round_to), round(turkish_all_f1_score_token_6[1], round_to))
    print('turkish_all_f1_score_token_7', round(turkish_all_f1_score_token_7[0], round_to), round(turkish_all_f1_score_token_7[1], round_to))
    print('turkish_all_f1_score_token_8', round(turkish_all_f1_score_token_8[0], round_to), round(turkish_all_f1_score_token_8[1], round_to))

print('english_all_f1_score_sequence_macro', round(english_all_f1_score_sequence_macro[0], round_to), round(english_all_f1_score_sequence_macro[1], round_to))
print('english_all_f1_score_sequence_0', round(english_all_f1_score_sequence_0[0], round_to), round(english_all_f1_score_sequence_0[1], round_to))
print('english_all_f1_score_sequence_1', round(english_all_f1_score_sequence_1[0], round_to), round(english_all_f1_score_sequence_1[1], round_to))
print('english_all_f1_score_sequence_2', round(english_all_f1_score_sequence_2[0], round_to), round(english_all_f1_score_sequence_2[1], round_to))
print('english_all_f1_score_token_macro', round(english_all_f1_score_token_macro[0], round_to), round(english_all_f1_score_token_macro[1], round_to))
print('english_all_f1_score_token_0', round(english_all_f1_score_token_0[0], round_to), round(english_all_f1_score_token_0[1], round_to))
print('english_all_f1_score_token_1', round(english_all_f1_score_token_1[0], round_to), round(english_all_f1_score_token_1[1], round_to))
print('english_all_f1_score_token_2', round(english_all_f1_score_token_2[0], round_to), round(english_all_f1_score_token_2[1], round_to))
print('english_all_f1_score_token_3', round(english_all_f1_score_token_3[0], round_to), round(english_all_f1_score_token_3[1], round_to))
print('english_all_f1_score_token_4', round(english_all_f1_score_token_4[0], round_to), round(english_all_f1_score_token_4[1], round_to))
if TAG_TYPE == 'BIO':
    print('english_all_f1_score_token_5', round(english_all_f1_score_token_5[0], round_to), round(english_all_f1_score_token_5[1], round_to))
    print('english_all_f1_score_token_6', round(english_all_f1_score_token_6[0], round_to), round(english_all_f1_score_token_6[1], round_to))
    print('english_all_f1_score_token_7', round(english_all_f1_score_token_7[0], round_to), round(english_all_f1_score_token_7[1], round_to))
    print('english_all_f1_score_token_8', round(english_all_f1_score_token_8[0], round_to), round(english_all_f1_score_token_8[1], round_to))

