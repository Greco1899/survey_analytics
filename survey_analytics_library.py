
# imports
import pandas as pd
import re



# replace text with multiple replacements
def replace_text(string, dict_of_replacements):
    '''
    replace multiple substrings in a string with a dictionary of replacements
    to be used if replacements are fixed and do not require regex as replace() is faster than re.sub()
    for regex replacements use clean_text()
    arguments:
    string (str): string for replacement
    dict_of_replacements (dict): dictionary of substring to replace and replacement
        e.g. {'to replace this':'with this',...}
    returns:
    a string with substrings replaced
    '''
    # loop through dict
    for key, value in dict_of_replacements.items():
        # perform replacement
        string = string.replace(key, value)
    # return
    return string



# clean text string
def clean_text(text_string, list_of_replacements, lowercase=True, ignorecase=False):
    '''
    clean text string
        lower case string
        regex sub user defined patterns with user defined replacements
    
    arguments:
    text_string (str): text string to clean
    list_of_replacements (list): a list of tuples consisting of regex pattern and replacement value
        e.g. [('[^a-z\s]+', ''), ...]
    lowercase (bool): default to True, if True, convert text to lowercase
    ignorecase (bool): default to False, if True, ignore case when applying re.sub()
        
    returns:
    a cleaned text string 
    '''
    
    # check lowercase argument
    if lowercase:
        # lower case text string 
        clean_string = text_string.lower()
    else:
        # keep text as is
        clean_string = text_string
    
    if ignorecase:
        # loop through each pattern and replacement
        for pattern, replacement in list_of_replacements:
            # replace defined pattern with defined replacement value
            clean_string = re.sub(pattern, replacement, clean_string, flags=re.IGNORECASE)
    else:
        # loop through each pattern and replacement
        for pattern, replacement in list_of_replacements:
            # replace defined pattern with defined replacement value
            clean_string = re.sub(pattern, replacement, clean_string)        
        
    # return
    return clean_string




# convert transformer model zero shot classification prediction into dataframe
def convert_zero_shot_classification_output_to_dataframe(model_output):
    '''
    convert zero shot classification output to dataframe
    model's prediction is a list dictionaries
    e.g. each prediction consists of the sequence being predicted, the user defined labels,
        and the respective scores.
    [
    {'sequence': 'the organisation is generally...',
     'labels': ['rewards', 'resourcing', 'leadership'],
     'scores': [0.905086100101471, 0.06712279468774796, 0.027791114524006844]}, 
    ...
        ]
    the function pairs the label and scores and stores it as a dataframe
    it also identifies the label with the highest score
    
    arguments:
    model_output (list): output from transformer.pipeline(task='zero-shot-classification')
    
    returns:
    a dataframe of label and scores for each prediction
    
    '''
    
    # store results as dataframe
    results = pd.DataFrame(model_output)
    # zip labels and scores as dictionary
    results['labels_scores'] = results.apply(lambda x: dict(zip(x['labels'], x['scores'])), axis=1) 
    # convert labels_scores to dataframe
    labels_scores = pd.json_normalize(results['labels_scores'])
    # get label of maximum score as new column
    labels_scores['label'] = labels_scores.idxmax(axis=1)
    # get score of maximum score as new column
    labels_scores['score'] = labels_scores.max(axis=1)
    # concat labels_scores to results
    results = pd.concat([results, labels_scores], axis=1)
    # drop unused columns
    results = results.drop(['labels', 'scores'], axis=1)

    # return
    return results


# convert transformer model sentiment classification prediction into dataframe
def convert_sentiment_classification_output_to_dataframe(text_input, model_output):
    '''
    convert sentiment classification output into a dataframe

    the model used distilbert-base-uncased-finetuned-sst-2-english outputs a list of lists with two dictionaries,
    within each dictionary is a label negative or postive and the respective score
    [
        [
            {'label': 'NEGATIVE', 'score': 0.18449656665325165},
            {'label': 'POSITIVE', 'score': 0.8155034780502319}
            ],
            ...
    ]
    the scores sum up to 1, and we extract only the positive score in this function,
    append the scores to the model's input and return a dataframe

    arguments:
    text_input (list): a list of sequences that is input for the model
    model_output (list): a list of labels and scores

    return:
    a dataframe of sequences and sentiment score

    '''
    # store model positive scores as dataframe
    results = pd.DataFrame(model_output)[[1]]
    # get score from column
    results = results[1].apply(lambda x: x.get('score'))
    # store input sequences and scores as dataframe
    results = pd.DataFrame({'sequence':text_input, 'score':results})
    
    # return
    return results