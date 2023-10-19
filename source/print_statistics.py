import pandas as pd

try:
    from read_data import read_combined_data
    from commons import *
except ModuleNotFoundError:
    from source.read_data import read_combined_data
    from source.commons import *

def print_number_of_tweets(data):
    print('Number of tweets:', len(data))

def print_number_of_neutral_tweets(data):
    print('Number of neutral tweets:', (data['label'] == LABEL_DICTIONARY['neutral']).sum())

def print_number_of_offensive_tweets(data):
    print('Number of offensive tweets:',(data['label'] == LABEL_DICTIONARY['offensive']).sum())

def print_number_of_hateful_tweets(data):
    print('Number of hateful tweets:', (data['label'] == LABEL_DICTIONARY['hateful']).sum())

def print_number_of_tie_tweets(data):
    print('Number of tie tweets:', (data['label'] == ANNOTATION_NON_MAJOR).sum())

def print_number_of_skipped_tweets(data):
    print('Number of skipped tweets:', (data['label'] == ANNOTATION_SKIPPED).sum())

def print_number_of_tweets_with_hashtag(data):
    tweets_with_hashtag = data['text'].str.contains('#')
    print('Number of tweets with hashtag:', tweets_with_hashtag.sum())
    print('Average hashtag count per hashtag tweet:', round(data[tweets_with_hashtag]['text'].str.count('#').mean()))

def print_number_of_tweets_with_url(data):
    tweets_with_url = data['text'].str.contains('http')
    print('Number of tweets with url:', tweets_with_url.sum())
    print('Average url count per url tweet:', round(data[tweets_with_url]['text'].str.count('http').mean()))

def print_number_of_tweets_with_emoji(data):
    tweets_with_emoji = data['text'].apply(count_emojis).astype(bool)
    print('Number of tweets with emoji:', tweets_with_emoji.sum())
    print('Average emoji count per emoji tweet:', round(data[tweets_with_emoji]['text'].apply(count_emojis).mean()))

def print_first_and_last_tweet_date(data):
    # example date: Tue Oct 13 21:23:32 +0000 2009
    date_time_format = '%a %b %d %H:%M:%S %z %Y'
    date = pd.to_datetime(data['created_at'], format=date_time_format) # date or created_at
    print('First tweet date:', date.min())
    print('Last tweet date:', date.max())

def print_tweet_length(data):
    # average
    print('Average tweet length:', round(data['text'].str.len().mean()))
    # max
    print('Max tweet length:', data['text'].str.len().max())
    # min
    print('Min tweet length:', data['text'].str.len().min())

def print_tweet_word_count(data):
    # average
    print('Average tweet word count:', round(data['text'].str.split().str.len().mean()))
    # max
    print('Max tweet word count:', data['text'].str.split().str.len().max())
    # min
    print('Min tweet word count:', data['text'].str.split().str.len().min())

def print_number_of_users(data):
    print('Number of users:', len(data['user_id'].unique()))    

def print_annotator_count(data):
    print('Number of tweets annotated by 2:', (data['annotated_by_how_many_annotators'] == 2).sum())
    print('Number of tweets annotated by 4:', (data['annotated_by_how_many_annotators'] == 4).sum())

def print_span_statistics(data):
    data = data.fillna('')
    spans = MajorityData.encoded_span_string_dataframe2span_list(data)
    span_statistics = [x.calculate_statistics() for x in spans]
    
    # print the number of tweets with HTARs, HSIGs, OTARs, OSIGs
    n_htar_tweets = len([x for x in span_statistics if 'HTAR' in x[0]])
    print('Number of tweets with HTARs:', n_htar_tweets)
    n_hsig_tweets = len([x for x in span_statistics if 'HSIG' in x[0]])
    print('Number of tweets with HSIGs:', n_hsig_tweets)
    n_otar_tweets = len([x for x in span_statistics if 'OTAR' in x[0]])
    print('Number of tweets with OTARs:', n_otar_tweets)
    n_osig_tweets = len([x for x in span_statistics if 'OSIG' in x[0]])
    print('Number of tweets with OSIGs:', n_osig_tweets)
    
    # to prevent division by 0
    if n_htar_tweets == 0:
        n_htar_tweets = 1
    if n_hsig_tweets == 0:
        n_hsig_tweets = 1
    if n_otar_tweets == 0:
        n_otar_tweets = 1
    if n_osig_tweets == 0:
        n_osig_tweets = 1
    
    # print the average number of HTARs, HSIGs, OTARs, OSIGs per tweet
    average_htar_interval_count = round(sum([x[1][x[0].index('HTAR')] for x in span_statistics if 'HTAR' in x[0]]) / n_htar_tweets)
    print('Average number of HTARs per tweet:', average_htar_interval_count)
    average_hsig_interval_count = round(sum([x[1][x[0].index('HSIG')] for x in span_statistics if 'HSIG' in x[0]]) / n_hsig_tweets)
    print('Average number of HSIGs per tweet:', average_hsig_interval_count)
    average_otar_interval_count = round(sum([x[1][x[0].index('OTAR')] for x in span_statistics if 'OTAR' in x[0]]) / n_otar_tweets)
    print('Average number of OTARs per tweet:', average_otar_interval_count)
    average_osig_interval_count = round(sum([x[1][x[0].index('OSIG')] for x in span_statistics if 'OSIG' in x[0]]) / n_osig_tweets)
    print('Average number of OSIGs per tweet:', average_osig_interval_count)
    
    # print the average length of HTARs, HSIGs, OTARs, OSIGs in words
    average_htar_len_in_words =  round(sum([x[2][x[0].index('HTAR')] for x in span_statistics if 'HTAR' in x[0]]) / n_htar_tweets)
    print('Average length of HTARs in words:', average_htar_len_in_words)
    average_hsig_len_in_words =  round(sum([x[2][x[0].index('HSIG')] for x in span_statistics if 'HSIG' in x[0]]) / n_hsig_tweets)
    print('Average length of HSIGs in words:', average_hsig_len_in_words)
    average_otar_len_in_words =  round(sum([x[2][x[0].index('OTAR')] for x in span_statistics if 'OTAR' in x[0]]) / n_otar_tweets)
    print('Average length of OTARs in words:', average_otar_len_in_words)
    average_osig_len_in_words =  round(sum([x[2][x[0].index('OSIG')] for x in span_statistics if 'OSIG' in x[0]]) / n_osig_tweets)
    print('Average length of OSIGs in words:', average_osig_len_in_words)

def main():
    data = read_combined_data()
    # language: 0 turkish, 1 english
    # topic: 0 religion, 1 gender, 2 race, 3 politics, 4 sport
    # label: -2 skipped, -1 tie, 0 neutral, 1 offensive, 2 hateful
    
    print("----------------------------------------DISTRIBUTION----------------------------------------")
    language = (0, 1)
    topic = (0, 1, 2, 3, 4)
    label = (-2, -1, 0, 1, 2)
    for lang in language:
        print('-----Language:', lang)
        for top in topic:
            print('---Topic:', top)
            for lab in label:
                print('-Label:', lab)
                print_number_of_tweets(data[(data['language'] == lang) & (data['topic'] == top) & (data['label'] == lab)])
                
    print("----------------------------------------LABELS----------------------------------------")
    for lang in language:
        print('-----Language:', lang)
        print_number_of_tweets(data[(data['language'] == lang)])
        print_number_of_neutral_tweets(data[(data['language'] == lang)])
        print_number_of_offensive_tweets(data[(data['language'] == lang)])
        print_number_of_hateful_tweets(data[(data['language'] == lang)])
        print_number_of_tie_tweets(data[(data['language'] == lang)])
        print_number_of_skipped_tweets(data[(data['language'] == lang)])
        print_number_of_tweets_with_hashtag(data[(data['language'] == lang)])
        print_number_of_tweets_with_url(data[(data['language'] == lang)])
        print_number_of_tweets_with_emoji(data[(data['language'] == lang)])
        print_first_and_last_tweet_date(data[(data['language'] == lang)])
        print_tweet_length(data[(data['language'] == lang)])
        print_tweet_word_count(data[(data['language'] == lang)])
        print_number_of_users(data[(data['language'] == lang)])
        print_annotator_count(data[(data['language'] == lang)])
    
    print("----------------------------------------ANNOTATION----------------------------------------")
    for lang in language:
        print('-----Language:', lang)
        for lab in label[2:]:
            print('---Label:', lab)
            print_span_statistics(data[(data['language'] == lang) & (data['label'] == lab)])
            

if __name__ == '__main__':
    main()
    #quit()
    