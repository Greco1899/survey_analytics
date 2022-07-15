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

# factor analysis
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats import zscore

# nlp
from bertopic import BERTopic

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
def read_tweet_data():
    tokyo = pd.read_csv(data_path+'tokyo_olympics_tweets.csv')
    return tokyo
tokyo = read_tweet_data()

@st.cache(allow_output_mutation=True)
def load_bertopic_model_unclean():
    topic_model = BERTopic.load(model_path+'bertopic_model_tokyo_olympics_tweets_unclean')
    return topic_model
topic_model_unclean = load_bertopic_model_unclean()

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

# plot topics using unclean data
fig = LIB.visualize_barchart_titles(
    topic_model=topic_model_unclean,
    subplot_titles=None,
    n_words=5,
    top_n_topics=8,
    height=300
)
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

# plot topics using clean data with stopwords removed
fig = LIB.visualize_barchart_titles(
    topic_model=topic_model,
    subplot_titles=labelled_topics,
    n_words=5,
    top_n_topics=8,
    height=300
)
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
    inspect_topic = st.number_input(f'Enter Topic (from {first_topic} to {last_topic}) to Inspect:', min_value=first_topic, max_value=last_topic, value=8)
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
st.write('''
    
    ''')
st.write('\n')
