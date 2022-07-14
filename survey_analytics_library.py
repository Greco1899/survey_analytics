
# imports
import pandas as pd
import numpy as np
import streamlit as st
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import zipfile
from xml.etree.cElementTree import XML

import re
from nltk.corpus import stopwords



# # create elbow plot with kmeans to find optimal number of clusters
# def create_elbow_plot_kmeans(df, num_clusters, init_method='k-means++', n_init=10, random_state=42, plot=True, template='simple_white', save=False):
#     '''
#     create elbow plot with kmeans to find optimal number of clusters based on inertia
#     where the clusters strikes a balance between being not segmented enough and being too fragmented
    
#     we look for the point of diminishing returns (also known as the 'elbow') in terms of the inertia,
#     where inertia is how close the data points are to their respective centers or centroids
    
#     arguments:
#     df (df): a dataframe of data to cluster
#     num_clusters (int): number of clusters to plot
#     init_method (str): default to 'k-means++', other option is 'random'
#     n_init (int): default to 10, number of times to run model, cost from the best run will be used
#     random_state (int): default to 42, random seed used to initialise the model
#     plot (bool): default to True, option to turn off plots
#     template (str): default to 'simple_white', change as desired
#     save (bool): default to False, if True save plot as .html file

#     returns:
#     a list of inertia for each run
#     '''

#     # create empty list to store inertia for each run
#     inertia = []
#     # define range of clusters to try
#     k = range(2, num_clusters+1)

#     # loop through number of clusters
#     for num_clusters in tqdm(k):
#         # define model
#         kmeans = KMeans(n_clusters=num_clusters, init=init_method, n_init=n_init, random_state=random_state)
#         # fit and predict data
#         kmeans.fit_predict(df)
#         # get predicted labels
#         predicted_labels = kmeans.labels_
#         # append score to list of scores
#         inertia.append(kmeans.inertia_)

#     # plot elbow plot
#     if plot:
#         fig = px.line(
#             pd.DataFrame({'num_clusters':list(k), 'inertia':inertia}), 
#             x='num_clusters', 
#             y='inertia',
#             title='Elbow Plot for Optimal Number of Clusters with '+init_method,
#             markers=True,
#             template=template,
#             width=800,
#             height=500,
#             )
#         st.plotly_chart(fig, use_container_width=True)
#         if save:
#             fig.write_html('Elbow Plot for Optimal Number of Clusters with '+init_method+'.html')
    
#     # return
#     return inertia



# # create plot of silhouette scores with sklearn model to find optimal number of clusters
# def silhouette_score_plot_kmeans(df, num_clusters, init_method='k-means++', n_init=10, random_state=42, plot=True, template='simple_white', save=False):
#     '''
#     create plot of silhouette score with kmeans to find optimal number of clusters
#     where the clusters strikes a balance between being not segmented enough and being too fragmented
#     the closer the score is to 1, the more easily distinguishable are the clusters from each other
    
#     arguments:
#     df (df): a dataframe of data to cluster
#     num_clusters (int): number of clusters to plot
#     init_method (str): default to 'k-means++', other option is 'random'
#     n_init (int): default to 10, number of times to run model, cost from the best run will be used
#     random_state (int): default to 42, random seed used to initialise the model
#     plot (bool): default to True, option to turn off plots
#     template (str): default to 'simple_white', change as desired
#     save (bool): default to False, if True save plot as .html file

#     returns:
#     a list of silhouette scores for each run
#     '''

#     # create empty list to store silhoutte scores for each run
#     silhouette_scores = []
#     # define range of clusters to try
#     k = range(2, num_clusters+1)

#     # loop through number of clusters
#     for num_clusters in tqdm(k):
#         # define model
#         kmeans = KMeans(n_clusters=num_clusters, init=init_method, n_init=n_init, random_state=random_state)
#         # fit and predict data
#         kmeans.fit_predict(df)
#         # get predicted labels
#         predicted_labels = kmeans.labels_
#         # get silhoutte score
#         score = silhouette_score(df, predicted_labels)
#         # append score to list of scores
#         silhouette_scores.append(score)
        
#     # plot silhouette scores
#     if plot:
#         fig = px.line(
#             pd.DataFrame({'num_clusters':list(k), 'silhouette_scores':silhouette_scores}), 
#             x='num_clusters', 
#             y='silhouette_scores',
#             title='Silhouette Scores for Optimal Number of Clusters with '+init_method,
#             markers=True,
#             template=template,
#             width=800,
#             height=500,
#             )
#         st.plotly_chart(fig, use_container_width=True)
#         if save:
#             fig.write_html('Silhouette Scores for Optimal Number of Clusters with '+init_method+'.html')
    
#     # return
#     return silhouette_scores



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



# remove stopwords from tokens
def remove_stopwords(tokens, language='english'):
    '''
    remove stopwords from tokens using list comprehension
    default to using english stopwords
    arguments:
    tokens (list): list of token#s, output of word_tokenize()
    language (str): default to english
    returns:
    a list of tokens without stopwords
    '''
    # define stopwords and store as a set
    stopwords_set = set(stopwords.words(language))
    # check if word is in list of stopwords
    # returns a list of words not found in list of stopwords
    stopwords_removed = [word for word in tokens if word not in stopwords_set]
    # return
    return stopwords_removed



import itertools
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def visualize_barchart_titles(topic_model,
                       topics: List[int] = None,
                       subplot_titles: List[str] = None,
                       top_n_topics: int = 8,
                       n_words: int = 5,
                       width: int = 250,
                       height: int = 250) -> go.Figure:
    """ Visualize a barchart of selected topics

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most frequent topics.
        n_words: Number of words to show in a topic
        width: The width of each figure.
        height: The height of each figure.

    Returns:
        fig: A plotly figure

    Usage:

    To visualize the barchart of selected topics
    simply run:

    ```python
    topic_model.visualize_barchart()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_barchart()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/bar_chart.html"
    style="width:1100px; height: 660px; border: 0px;""></iframe>
    """
    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list()[0:6])

    # Initialize figure
    if subplot_titles is None:
        subplot_titles = [f"Topic {topic}" for topic in topics]
    else:
        subplot_titles = subplot_titles
    columns = 4
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.4 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles)

    # Add barchart for each topic
    row = 1
    column = 1
    for topic in topics:
        words = [word + "  " for word, _ in topic_model.get_topic(topic)][:n_words][::-1]
        scores = [score for _, score in topic_model.get_topic(topic)][:n_words][::-1]

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': "<b>Topic Word Scores",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width*4,
        height=height*rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig



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