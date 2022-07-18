# launch app
# streamlit run "survey_analytics_streamlit.py"

# imports
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle

# factor analysis
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats import zscore

# nlp
from bertopic import BERTopic
from transformers import pipeline

# custom
import survey_analytics_library as LIB

# st.set_page_config(layout='wide')

# define data file path
data_path = 'data' + os.sep
# define model file path
model_path = 'models' + os.sep

# load and cache all data and models to improve app performance
@st.cache
def read_survey_data():
    data_survey = pd.read_csv(data_path+'bfi_sample_answers.csv')
    data_questions = pd.read_csv(data_path+'bfi_sample_questions.csv')
    return data_survey, data_questions
data_survey, data_questions = read_survey_data()

@st.cache
def read_tokyo_data():
    tokyo = pd.read_csv(data_path+'tokyo_olympics_tweets.csv')
    return tokyo
tokyo = read_tokyo_data()

@st.cache(allow_output_mutation=True)
def load_bertopic_model():
    topic_model = BERTopic.load(model_path+'bertopic_model_tokyo_olympics_tweets')
    return topic_model
topic_model = load_bertopic_model()

@st.cache
def read_topic_results():
    topic_results = pd.read_csv(data_path+'topic_results.csv')
    return topic_results
topic_results = read_topic_results()

@st.cache
def read_climate_change_results():
    sentiment_results = pd.read_csv(data_path+'sentiment_results.csv')
    zero_shot_results = pd.read_csv(data_path+'zero_shot_results.csv')
    return sentiment_results, zero_shot_results
sentiment_results, zero_shot_results = read_climate_change_results()


# write title of app
st.title('DACoP - Survey Analytics')
st.markdown('''---''')



st.header('Clustering Survey Responders')
st.write('''
    Having knowledge about different groups of responders can help us to customise our interactions with them.  
    E.g. Within the Financial Institutions we have banks, insurers, and payment services.   
    We want to be able to cluster survey reponders into various groups based on how their answers.  
    This can be achieved though **Factor Analysis**.   
    ''')
st.write('\n')
st.write('\n')

# copy data
df_factor_analysis = data_survey.copy()

st.subheader('Sample Survey Data')
st.write('''
    Here we have a sample survey dataset where responders answer questions about their personality traits on a scale from 1 (Very Inaccurate) to 6 (Very Accurate).  
    Factor Analysis gives us \'factors\' or groups of responders into groups can provide us insights about the different personalities of the responders.  
    ''')

# split page into two columns
# display survey questions and answers as dataframes side by side
col1, col2 = st.columns(2)
with col1:
    st.write('Survey Questions')
    st.dataframe(data_questions)
with col2:
    st.write('Survey Answers')
    st.dataframe(df_factor_analysis)
st.write('\n')
st.write('\n')

st.subheader('Factor Analysis Suitability')
st.write('''
    Before performing Factor Analysis on the data, we need to evaluate if it is suitable to do so.  
    We apply two statistical tests (Bartlett's and KMO test) the data. 
    ''')

# interactive button to run statistical test to determine suitability for factor analysis
if st.button('Run Tests'):
    # test with the null hypothesis that the correlation matrix is an identity matrix
    bartlett_sphericity_stat, p_value = calculate_bartlett_sphericity(x=df_factor_analysis)
    # test how predictable of a variable by others
    kmo_per_variable, kmo_total = calculate_kmo(x=df_factor_analysis)
    # print test results
    st.write(f'''
        The P Value from Bartlett\'s Test (suitability is less than 0.05): **{round(p_value, 2)}**  
        The Value from KMO Test (suitability is more than 0.60): **{round(kmo_total, 2)}**  
        ''')
    # set default status to 'Failed'
    fa_stat_test = 'Failed'
    # check if data passes both tests
    if p_value < 0.05 and kmo_total >= 0.6:
        fa_stat_test = 'Passed'

    st.success(f'Our data has **{fa_stat_test}** the two statistical tests!')

st.write('\n')
st.write('\n')

# define factor analyser model
fa = FactorAnalyzer()
# fit data
fa.fit(X=df_factor_analysis)

# get eigenvalues
eigenvalues, _ = fa.get_eigenvalues()
# get number of eigenvalues more than or equal to 1
optimal_factors = len([value for value in eigenvalues if value >= 1])
# store eigenvalues and number of clusters into a df for plotly
scree_df = pd.DataFrame({'Eigenvalues':eigenvalues, 'Number of Factors':list(range(1, len(eigenvalues)+1))}) 

st.subheader('Number of Clusters?')
st.write(f'''
    How many clusters or factors are appropriate for our data?  
    For Factor Analysis, we can determine the number of factors using the Kaiser criterion and a Scree Plot.  
    We should include factors with an Eigenvalue of at least 1.0.  
    ''')

# plot scree plot
fig = px.line(
    scree_df,
    x='Number of Factors', 
    y='Eigenvalues',
    markers=True,
    title='Scree Plot for Kaiser Criterion',
    template='simple_white',
    width=800,
    height=500,
    )
fig.add_hline(y=1, line_width=3, line_color='darkgreen')
st.plotly_chart(fig, use_container_width=True)
st.write(f'''
    Kaiser criterion is one of many guides to determine the number of factors, ultimately the decision on the number of factors to use is best decided by the user based on their use case.  
    ''')

# interactive form for user to enter different number of factors for analysis
with st.form('num_factor_form'):
    # define number input 
    user_num_factors = st.number_input('Enter desired number of factors:', min_value=1, max_value=10, value=6)
    # set factors to user input
    optimal_factors = user_num_factors
    # submit button for form to rerun app when user is ready
    submit = st.form_submit_button('Run Factor Analysis')

st.write('\n')
st.write('\n')

# define factor analyser model
fa = FactorAnalyzer(n_factors=optimal_factors, rotation='varimax')
# fit data
fa.fit(df_factor_analysis)
# generate factor loadings
loads_df = pd.DataFrame(fa.loadings_, index=df_factor_analysis.columns)

# fit and transform data
responder_factors = fa.fit_transform(df_factor_analysis)
# store results as df
responder_factors = pd.DataFrame(responder_factors)
# rename columns to 'factor_n'
responder_factors.columns = ['factor_'+str(col) for col in list(responder_factors)]
# use the max loading across all factors to determine a responder's cluster
responder_factors['cluster'] = responder_factors.apply(lambda s: s.argmax(), axis=1)

# define list of factor columns
list_of_factor_cols = [col for col in responder_factors.columns if 'factor_' in col]
st.subheader('Fator Analysis Results')
st.write('''
    Factor analysis gives us a loading for every factor for each responder.  
    We assign each responder to a factor or cluster based on their maximum loading across all the factors.  
    ''')
# highlight factor with max loadings
st.dataframe(responder_factors.style.highlight_max(axis=1, subset=list_of_factor_cols, props='color:white; background-color:green;').format(precision=2))
st.write('\n')

# count number of responders in each cluster
fa_clusters = df_factor_analysis.copy().reset_index(drop=True)
fa_clusters['cluster'] = responder_factors['cluster']
cluster_counts = fa_clusters['cluster'].value_counts().reset_index()
cluster_counts = cluster_counts.rename(columns={'index':'Cluster', 'cluster':'Count'})

# calculate z-scores for each cluster
fa_z_scores = df_factor_analysis.copy().reset_index(drop=True)
fa_z_scores = fa_z_scores.apply(zscore)
fa_z_scores['cluster'] = responder_factors['cluster']
fa_z_scores = fa_z_scores.groupby('cluster').mean().reset_index()
fa_z_scores = fa_z_scores.apply(lambda x: round(x, 2))

st.write('''
    Aggregating the scores of the clusters gives us detail insights to the personality traits of the responders.  
    The scores here have been normalised to Z-scores, a measure of how many standard deviations (SD) is the score away from the mean.  
    E.g. A Z-score of 0 indicates the score is identical to the mean, while a Z-score of 1 indicates the score is 1 SD away from the mean.  
    ''')
# define colour map for highlighting cells
cm = sns.light_palette('green', as_cmap=True)
# define list of question columns
list_of_question_cols = list(fa_z_scores.iloc[:,1:])
# display z-scores of clusters with conditional formatting
st.dataframe(fa_z_scores.style.background_gradient(cmap=cm, subset=list_of_question_cols).format(precision=2))
st.write('\n')

st.write('''
    Lastly, we can visualise the distribution of responders in each cluster.  
    ''')
# plot percentage of responders in each cluster
fig = px.pie(
    cluster_counts,
    values='Count', 
    names='Cluster', 
    hole=0.35,
    title='Percentage of Responders in Each Cluster',
    template='simple_white',
    width=1000,
    height=600,
    )
st.plotly_chart(fig, use_container_width=True)
st.markdown('''---''')






st.header('Uncovering Topics from Text Responses')
st.write('''
    With feedback forms or open-ended survey questions, we want to know what are the responders generally talking about.  
    One way would be to manually read all the collected response to get a sense of the topics within, however, this is very manual and subjective.  
    Using **Topic Modelling**, we can programmatically extract common topics with the help of machine learning.  
    ''')
st.write('\n')

st.write(f'''
    Here we have {len(tokyo):,} tweets from the Tokyo Olympics, going through them manually and coming up with topics would not be practical.  
    ''')
# rename column
tokyo = tokyo.rename(columns={'text':'Tweet'})
# display raw tweets
st.dataframe(tokyo)
st.write('\n')
st.write('\n')

st.write('''
    Lets generate some topics without performing any cleaning to the data.  
    ''')
st.write('\n')

# load and plot topics using unclean data
with open('data/topics_tokyo_unclean.pickle', 'rb') as pkl:
    fig = pickle.load(pkl)
st.plotly_chart(fig, use_container_width=True)

st.write('''
    From the chart above, we can see that 'Topic 0' and 'Topic 5' have some words that are not as meaningful.  
    For 'Topic 0', we already know that the tweets are about the Tokyo 2020 Olympics, having a topic for that isn't helpful.  
    'Tokyo', '2020', 'Olympics', etc., we refer to these as *stopwords*, and lets remove them and regenerate the topics.  
    ''')
st.write('\n')

# define manually created topic labels
labelled_topics = [
    'Barbra Banda (Zambian Footballer)',
    'Indian Pride',
    'Sutirtha Mukherjee (Indian Table Tennis Player)',
    'Mirabai Chanu (Indian Weightlifter)',
    'Road Race',
    'Japan Volleyball',
    'Sam Kerr (Australian Footballer)',
    'Vikas Krishan (Indian Boxer)',
    ]

# load plot topics using clean data with stopwords removed
with open('data/topics_tokyo.pickle', 'rb') as pkl:
    fig = pickle.load(pkl)
st.plotly_chart(fig, use_container_width=True)

st.write('''
    Now we can see that the topics have improved.  
    We can make use of the top words in each topic to come up with a meaningful name.  
    ''')
st.write('\n')
st.write('\n')

# store topic info as dataframe
topics_df = topic_model.get_topic_info()

st.write(f'''
    Next, we can also review the total number of topics and how many tweets are in each topic, to give us a sense of importance or priority.  
    There are a total of **{len(topics_df)-1}** topics, and the larget topic contains **{topics_df['Count'][1]}** tweets.  
    {topics_df['Count'][0]} tweets have also been assigned as Topic -1 or outliers. These tweets are unique compared to the others and there aren't enough of them to form a topic.   
    If there are too many or too few topics, there is also the option to further tune the model to refine the results. 
    ''')
# display topic info
st.dataframe(topics_df)
st.write('\n')

st.write('''
    One point to also note is that the machine is not only picking out keywords in a tweet to determine its topic.  
    The model has an understanding of the relationship between words, e.g. 'Andy Murray' is related to 'tennis'.  
    For example:  
    *'Cilic vs Menezes, after more than 3 hours and millions of unconverted match points, is one of the worst quality ten…'*  
    This tweet is in the Topic 9 - Tennis without the word 'tennis' in it.  

    Here we can inspect the individual tweets within each topic.  
    ''')

# define the first and last topic number
first_topic = topics_df['Topic'].iloc[0]
last_topic = topics_df['Topic'].iloc[-1]

# interative form for user to select a topic and inspect its top words and tweets
with st.form('inspect_tweets'):
    inspect_topic = st.number_input(f'Enter Topic (from {first_topic} to {last_topic}) to Inspect:', min_value=first_topic, max_value=last_topic, value=9)
    submit = st.form_submit_button('Inspect Topic')

# get top five words from list of tuples
inspect_topic_words = [i[0] for i in topic_model.get_topic(inspect_topic)[:5]]

st.write(f'''
    The top five words for Topic {inspect_topic} are: {inspect_topic_words}
    ''')
# display tweets from selected topic
st.dataframe(topic_results.loc[(topic_results['Topic'] == inspect_topic)])
st.markdown('''---''')






st.header('Classifiying Text Responses and Sentiment Analysis')
st.write(f'''
    With survey responses, sometimes as a business user, we already have an general idea of what responders are talking about and we want to categorise or classify the responses accordingly.  
    An an example, within the topic of 'Climate Change', we are interested in finance, politics, technology, and wildlife.  
    Using **Zero-shot Classification**, we can classify responses into one of these four categories.  
    As an added bonus, we can also find out how responders feel about the categories using **Sentiment Analysis**.  
    We'll use a different set of {len(sentiment_results):,} tweets related to climate change.  
    ''')
st.write('\n')

# rename column
sentiment_results = sentiment_results.rename(columns={'sequence':'Tweet'})
st.dataframe(sentiment_results[['Tweet']])

@st.cache(allow_output_mutation=True)
def load_transfomer_pipelines():
    classifier_zero_shot = pipeline(
        task='zero-shot-classification', 
        model='valhalla/distilbart-mnli-12-1',
        return_all_scores=True
        )
    classifier_sentiment = pipeline(
        task='sentiment-analysis', 
        model = 'distilbert-base-uncased-finetuned-sst-2-english',
        return_all_scores=True
        )
    return classifier_zero_shot, classifier_sentiment
classifier_zero_shot, classifier_sentiment = load_transfomer_pipelines()

# define candidate labels
candidate_labels = [
    'finance',
    'politics',
    'technology',
    'wildlife',
]

# define sample tweet
sample_tweet_index = 5000

# define the first and last topic number
# create range of index
tweet_index = sentiment_results.index
first_tweet = tweet_index[0]
last_tweet = tweet_index[-1]

st.write(f'''
    As a demonstration, we'll define some categories and pick a tweet to classify and determine its sentiment.  
    Feel free to add your own categories or even input your own text!  
    ''')

# interactive input for user to define candidate labels and tweet index for analysis
with st.form('classify_tweets'):
    # input for labels
    user_defined_labels = st.text_input('Enter categories (separate categories by comma):', ', '.join(candidate_labels))
    candidate_labels = user_defined_labels
    # input for tweet index
    user_define_tweet = st.number_input(f'Enter tweet index (from {first_tweet} to {last_tweet}) to classify:', min_value=first_tweet, max_value=last_tweet, value=sample_tweet_index)
    sample_tweet_index = user_define_tweet
    sample_tweet = sentiment_results['Tweet'].iloc[sample_tweet_index]
    # input for user defined text
    user_defined_input = st.text_input('Enter custom text (optional, leave blank to use Tweets):', '')
    # check if user has entered any custom text
    # if user_define_input is not blank, then override sample_tweet
    if user_defined_input:
        sample_tweet = user_defined_input

    # submit form
    submit = st.form_submit_button('Classify Tweet')

st.write('\n')
st.write(f'''
    Here are the results:  
    ''')
st.write(f'Input Text: *\'{sample_tweet}\'*')

# get predictions from models
zero_shot_sample = classifier_zero_shot(sample_tweet, candidate_labels)
sentiment_sample = classifier_sentiment(sample_tweet)

# get sentiment
sentiment_sample = sentiment_sample[1].get('score')
sentiment_label = 'positive'
if sentiment_sample < 0.5:
    sentiment_label = 'negative'

st.write(f'''
    The main category is: **{zero_shot_sample['labels'][0]}** with a score of {round(zero_shot_sample['scores'][0], 2)}  
    Main category score ranges from 0 to 1, with 1 being very likely.  

    The full set of scores are: {dict(zip(zero_shot_sample['labels'], [round(score, 2) for score in zero_shot_sample['scores']]))}  
    Full set of scores cores add up to 1.    
    
    The sentiment is: **{sentiment_label}** with a score of {round(sentiment_sample, 2)}  
    Sentiment score ranges from 0 to 1, with 1 being very positive.  
    ''')
st.write('\n')
st.write('\n')

# drop unused columns and rename columns
zero_shot_results = zero_shot_results.drop('labels_scores', axis=1)
zero_shot_results = zero_shot_results.rename(columns={'sequence':'tweet', 'label':'category'})
st.write(f'''
    Lets review all the tweets and how they fall into the categories of finance, politics, technology, and wildlife.  
    ''')

st.dataframe(zero_shot_results)

st.write(f'''
    We can observe that the model does not have strong confidence in predicting the categories for some of the tweets.  
    It is likely that the tweet does not natually fall into one of the defined categories.  
    Before performing further analysis on our results, we can set a score threshold to only keep predictions that we're confident in.  
    ''')
st.write('\n')

# interactive input for user to define candidate labels and tweet index for analysis
with st.form('classification_score_threshold'):
    user_defined_threshold = st.number_input('Enter score threshold (between 0.01 and 0.99):', min_value=0.01, max_value=0.99, value=0.7, step=0.05)
    # submit form
    submit = st.form_submit_button('Set Threshold')
st.write('\n')

# filter and keep results with score above defined threshold
zero_shot_results_clean = zero_shot_results.loc[(zero_shot_results['score'] >= user_defined_threshold)].copy()

# rename columns
sentiment_results.columns = ['tweet', 'sentiment']

st.write(f'''
    The predictions get better with a higher threshold, but reduces the final number of tweets available for further analysis.  
    Out of the 10,000 tweets, we are now left with {len(zero_shot_results_clean)}.  
    We also add on the sentiment score for the tweets, the score here ranges from 0 (most negative) to 1 (most positive).  
    ''')

# merge in sentiment score on index
# drop unused columns
classification_sentiment_df = pd.merge(zero_shot_results_clean, sentiment_results[['sentiment']], how='left', left_index=True, right_index=True)
classification_sentiment_df = classification_sentiment_df[['tweet', 'category', 'score', 'sentiment']]
st.dataframe(classification_sentiment_df)

st.write(f'''
    The difficult part for zero-shot classification is defining the right set of categories for each business case.  
    Some trial and error is required to find the appropriate words that can return the optimal results.   
    ''')
st.write('\n')

# group by category, count tweets and get mean of sentiment
classification_sentiment_agg = classification_sentiment_df.groupby(['category']).agg({'tweet':'count', 'sentiment':'mean'}).reset_index()
classification_sentiment_agg = classification_sentiment_agg.rename(columns={'tweet':'count'})

st.write(f'''
    Finally, we can visualise the percentage of tweets in each category and the respective average sentiment scores.  
    ''')

fig = px.pie(
    classification_sentiment_agg,
    values='count',
    names='category',
    hole=0.35,
    title='Percentage of Tweets in Each Category',
    template='simple_white',
    width=1000,
    height=600
)
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig)

fig = px.bar(
    classification_sentiment_agg,
    x='category',
    y='sentiment',
    title='Average Sentiment of Tweets in Each Category <br><sup>Overall, the sentiment of the tweets are on the negative side.</sup>',
    template='simple_white',
    width=1000,
    height=600
)
fig.update_yaxes(range=[0, 1])
fig.add_hline(y=0.5, line_width=3, line_color='darkgreen')
st.plotly_chart(fig)

st.write('\n')
st.markdown('''---''')
