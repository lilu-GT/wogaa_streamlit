from gensim.models import Phrases
import gensim.corpora as corpora
import spacy
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize
from time import sleep, strftime
import os
from functools import reduce
import json
import io
import zipfile
import glob
import pandas as pd
import altair as alt
import streamlit as st
from os.path import basename
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS
from gensim.models.phrases import Phraser
from functional import seq
import numpy as np
import random
import docx
from docx.shared import Cm
from numpy.random import multinomial
from numpy import log, exp, argmax


random.seed(0)
np.random.seed(0)


def main():
    ############################################# Streamlit Setting #############################################
    st.set_page_config(
        # layout="wide",
        page_title="WOGAA Sentiment Analysis and Clustering")

    # Title
    st.title("WOGAA Tool for Feedback Analysis")

    # Perform sentiment analysis?
    # new_title = '<p style="vertical-align: text-bottom; color:Green; font-size: 22px;">New image</p>'
    # st.markdown(new_title, unsafe_allow_html=True)
    sentiment = st.radio(
        label='Do you want to perform sentiment correction?',
        options=('Yes, please', 'No, thanks'))

    if sentiment == 'Yes, please':
        sentiment_analysis = True
        st.success('Sentiment analysis will be performed. This may take a while.')
    else:
        sentiment_analysis = False
        st.success(
            "Sentiment analysis will not be performed.")

    # Check only Pos / only Neg / both
    st.text("")
    st.text("")
    high_option = "Only a high score feedback file, eg. ratings 4 and below"
    low_option = "Only a low score feedback file, eg. ratings 5 and above"
    both_option = "Both high and low score feedback files"
    high_low_file = st.radio(
        label='Which file/files do you want to perform the analysis?',
        options=(high_option, low_option, both_option))
    high_file = low_file = False
    if high_low_file == high_option:
        high_file = True
    if high_low_file == low_option:
        low_file = True
    if high_low_file == both_option:
        high_file = True
        low_file = True

    # Upload
    # --------------- High score Only --------------- #
    if high_file and (not low_file):
        # Sidebar
        st.sidebar.title("Input Field Names")
        utterance_field_high = st.sidebar.text_input(
            "Description Field (High score feedback file)", "What did you like most?").strip()
        # score_field_high = st.sidebar.text_input(
        #     "Score Field (High score feedback file)", "Rating").strip()

        # Upload
        st.text("")
        st.text("")
        uploaded_high_file = st.file_uploader(
            "Please upload a high score feedback file and fill in the input field names", type=['csv', 'xls', 'xlsx'])

        df_high = None

        if uploaded_high_file is not None:
            if uploaded_high_file.name.endswith('.csv'):
                df_high = pd.read_csv(uploaded_high_file)
            else:
                df_high = pd.read_excel(uploaded_high_file)
            columns_in_file = df_high.columns.to_list()
############################################# No Sentiment Correction #############################################
    if not sentiment_analysis:
        # =================== High score Only =================== #
        if df_high is not None:
            pos_neg = "positive_feedback"
            processed_tokens_key = "processed_tokens"
            sw_top_ratio = 0.02
            cluster_label = 'cluster_label'
            num_clusters = 40
            n_iters_1st = 30
            n_iters_2nd = 60
            num_top_words = 15  # try 20? for word cloud
            top_n_quote = 3
            cluster_num_ratio_list = [5, 4, 2]  # large
            cluster_size_list = ['Small', 'Medium', 'Large']
            # ----------------------- Preprocess for clustering -----------------------#

            df, df_empty_responses = preprocess_df(df_high, processed_tokens_key,
                                                   text_field=utterance_field_high,
                                                   sw_top_ratio=sw_top_ratio,
                                                   pos_neg=pos_neg)

            st.text('')
            st.subheader('Results:')
            # For 3 different sizes:
            for cluster_size, cluster_num_ratio in zip(cluster_size_list, cluster_num_ratio_list):
                df_clustered_output, df_empty_responses = perform_clustering(
                    processed_tokens_key, cluster_label, df, num_clusters, n_iters_1st, n_iters_2nd, cluster_num_ratio, columns_in_file, df_empty_responses)

                xlsx_file = to_excel(df_clustered_output, df_empty_responses)
                st.download_button(label=f'📥 Clustered result ({cluster_size} number of clusters)',
                                   data=xlsx_file,
                                   file_name=f'clustered_result_{cluster_size}.xlsx')

    else:
        score_field_high = st.sidebar.text_input(
            "Score Field (High score feedback file)", "Rating").strip()


def get_pos_neg_files(df_high, df_low):
    return df_high, df_low


@st.experimental_memo
def preprocess_df(df, processed_tokens_key, text_field, n_letters_stopwords=1, add_on_stopwords=[], allowed_postags=["NOUN", "ADJ", "VERB"], min_count_bigram=5, threshold_bigram=10, res_folder='../Result', dataset='', sw_top_ratio=0.02, pos_neg=''):
    # keep rows that are empty
    empty_responses_df = df[df[text_field].isnull()]
    df = df[~df[text_field].isnull()]

    data = df[text_field].str.lower().values.tolist()

    data_sent = [sent_tokenize(doc) for doc in data]

    data_pos_tags = pos_tagging(data_sent)

    if pos_neg == "negative_feedback":
        custom_stopwords = detect_stopwords(
            data_pos_tags, res_folder, dataset, sw_top_ratio, pos_neg=pos_neg)
    else:
        custom_stopwords = detect_stopwords(
            data_pos_tags, res_folder, dataset, sw_top_ratio, pos_neg=pos_neg)
    data_no_stopwords = remove_stopwords(
        data_pos_tags, custom_stopwords, add_on_stopwords, n_letters_stopwords)

    data_remove_pos = remove_pos_tags(data_no_stopwords, allowed_postags)

    data_lemma = lemmatize(data_remove_pos)

    bigram_model, trigram_model = build_ngram_model(
        data_lemma, min_count_bigram, threshold_bigram)
    data_ngrams = replace_ngrams(data_lemma, bigram_model, trigram_model)

    data_doc_tokens = combine_doc_tokens(data_ngrams)
    df[processed_tokens_key] = data_doc_tokens

    # drop rows that are empty before preprocessing
    df['word_count'] = df[text_field].str.split().str.len()
    df['word_count2'] = df[processed_tokens_key].str.len()

    empty_responses_df = pd.concat(
        [empty_responses_df, df[df['word_count'] <= 3]])
    df = df[df['word_count'] > 3]

    df['word_count2'] = df[processed_tokens_key].str.len()
    empty_responses_df = pd.concat(
        [empty_responses_df, df[df['word_count2'] < 1]]).reset_index(drop=True)
    df = df[df['word_count2'] >= 1].reset_index(drop=True)

    return df, empty_responses_df

#
# delete the folders
#


############### Clustering  ###############

@st.experimental_memo
def perform_clustering(processed_tokens_key, cluster_label, df, num_clusters, n_iters_1st, n_iters_2nd, cluster_num_ratio, columns_in_file, df_empty_responses):
    # ----------------------- Clustering -----------------------#
    # Round 1:
    gsdmm = GSDMM(processed_tokens_key, cluster_label)
    _, cluster_num_1st = gsdmm.cluster(
        df, num_clusters=num_clusters, n_iters=n_iters_1st, verbose=False)
    print(
        f"\n  First iteration: {cluster_num_1st} clusters predicted")
    cluster_num = round(cluster_num_1st/cluster_num_ratio)
    print(
        f"\n  Downscale {cluster_num_ratio} folds to have, {cluster_num} clusters for second iteration\n")

    # Round 2:
    df_clustered, cluster_num = gsdmm.cluster(
        df, num_clusters=cluster_num, n_iters=n_iters_2nd, verbose=False)
    df_clustered = df_clustered.sort_values(
        [cluster_label, "max_score"], ascending=[True, False])
    print(f"\n  {cluster_num} clusters predicted")
    print(
        f"  Clustered: {df_clustered.shape[0]}, Non-clustered: {df_empty_responses.shape[0]}")

    # ----------------------- Show and Download the results -----------------------#

    # Excel File
    columns_to_keep = columns_in_file + [cluster_label]
    df_clustered_output = df_clustered[columns_to_keep]
    df_empty_responses = df_empty_responses[columns_to_keep[:-1]]
    # st.text("Clustered feedback:")
    # st.write(df_clustered_output)

    return df_clustered_output, df_empty_responses

############### functions in preprocess_df ###############


def pos_tagging(data):
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    data_doc_sent = []
    for doc in data:
        doc_sent = []
        for sent in doc:
            sent = nlp(sent)
            doc_sent.append([(token.text, token.pos_, token.lemma_)
                            for token in sent])
        data_doc_sent.append(doc_sent)

    return data_doc_sent


def detect_stopwords(data_pos_tags, res_folder, dataset, sw_top_ratio, pos_neg):
    # combine tokens back to doc
    doc_list = []
    for doc in data_pos_tags:
        sent_tokens = []
        for sent in doc:
            for token in sent:
                sent_tokens.append(token[2])
        sent = ' '.join(sent_tokens)
        doc_list.append(sent)

    # IDF scores
    tf = TfidfVectorizer(use_idf=True, token_pattern='(?u)\\b\\w+\\b')
    tf.fit_transform(doc_list)
    idf = tf.idf_

    # map IDF scores to tokens
    token_idf = [idf[v] for k, v in tf.vocabulary_.items()]
    token = [k for k, v in tf.vocabulary_.items()]
    idf_df = pd.DataFrame(list(zip(token, token_idf)),
                          columns=['token', 'IDF']).sort_values("IDF").reset_index(drop=True)

    # remove tokens in spacy stopwords
    idf_df_no_sw = idf_df[~idf_df["token"].isin(SPACY_STOPWORDS)]

    # top sw_top_ratio% ratio of IDF tokens (from small to large)
    customised_stopwords = idf_df_no_sw.head(
        round(idf_df_no_sw.shape[0] * sw_top_ratio))

    custom_stopwords = list(customised_stopwords['token'])

    return custom_stopwords


def remove_stopwords(data, custom_stopwords, add_on_stopwords, n_letters_stopwords):
    stopwords_list = list(SPACY_STOPWORDS) + \
        custom_stopwords + add_on_stopwords
    return [
        [[token for token in sent if ((token[2] not in stopwords_list) and (
            len(token[0]) >= n_letters_stopwords))] for sent in doc]
        for doc in data
    ]


def remove_pos_tags(data, allowed_postags):
    return [
        [[token for token in sent if token[1] in allowed_postags]
            for sent in doc]
        for doc in data
    ]


def lemmatize(data):
    return [[[token[2] for token in sent] for sent in doc] for doc in data]


def build_ngram_model(data, min_count, threshold):
    docs = seq(data).flatten().to_list()
    bigram = Phrases(docs, min_count=min_count, threshold=threshold)
    trigram = Phrases(bigram[docs], min_count=min_count, threshold=threshold)

    bigram_model = Phraser(bigram)
    trigram_model = Phraser(trigram)

    return bigram_model, trigram_model


def replace_ngrams(data, bigram_model, trigram_model):
    return [[trigram_model[bigram_model[sent]] for sent in doc] for doc in data]


def combine_doc_tokens(data):
    data_doc_tokens = []

    for doc in data:
        doc_tokens = []
        for sent in doc:
            doc_tokens += sent
        data_doc_tokens.append(doc_tokens)

    return data_doc_tokens

############### GSDMM Class and functions ###############


class GSDMM:
    def __init__(self, processed_tokens_key, cluster_key):
        self.processed_tokens_key = processed_tokens_key
        self.cluster_key = cluster_key

    def cluster(self, df, num_clusters, n_iters=30, alpha=0.1, beta=0.1, verbose=True):

        id2word = corpora.Dictionary(df[self.processed_tokens_key])

        mgp = MovieGroupProcess(
            K=num_clusters, alpha=alpha, beta=beta, n_iters=n_iters, verbose=verbose)
        y_sampled = mgp.fit(df[self.processed_tokens_key],
                            vocab_size=len(id2word))

        # cluster prediction of max prob (no longer sampling)
        y = []
        p = []
        for doc in df[self.processed_tokens_key]:
            y_doc, p_norm_doc, p_doc = mgp.choose_best_label(doc)
            y.append(y_doc)
            p.append(p_norm_doc)

        df[self.cluster_key] = y

        # reset the cluster ids
        replacement_dict = {}
        c_counter = 0
        for c_id in set(y):
            replacement_dict[c_id] = c_counter
            c_counter += 1

        new_y = df[self.cluster_key].map(replacement_dict)
        df[self.cluster_key] = new_y
        df["max_score"] = p

        return df, c_counter

    def tfidf_keywords(self, df, num_keywords):

        keywords_df = pd.DataFrame()
        labels = sorted(list(df[self.cluster_key].unique()))

        # formatting: processed_tokens_key; tfidf: Count
        label_tokens = []
        for label in labels:

            temp_df = df[df[self.cluster_key] == label]
            sub_cat_tokens = " ".join(
                [
                    " ".join(row[self.processed_tokens_key])
                    for index, row in temp_df.iterrows()
                ]
            )
            label_tokens.append(sub_cat_tokens)

        # tfidf
        vectorizer = CountVectorizer()
        counts = vectorizer.fit_transform(label_tokens)

        # formatting into label-term matrix
        word_scores_df = pd.DataFrame(
            counts.todense().tolist(), columns=vectorizer.get_feature_names()
        )

        # extracting keywords
        keywords_df = pd.DataFrame()

        for index, row in word_scores_df.iterrows():
            label_df = pd.DataFrame(row.sort_values(
                ascending=False)).reset_index()
            label_df[self.cluster_key] = labels[index]
            label_df.columns = ["Term", "Frequency", self.cluster_key]

            label_df = label_df[:num_keywords]

            keywords_df = pd.concat([keywords_df, label_df])

        return keywords_df


class MovieGroupProcess:
    def __init__(self, verbose, K=8, alpha=0.1, beta=0.1, n_iters=30):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters
        self.verbose = verbose

        # slots for computed variables
        self.number_docs = None
        self.vocab_size = None
        self.cluster_doc_count = [0 for _ in range(K)]
        self.cluster_word_count = [0 for _ in range(K)]
        self.cluster_word_distribution = [{} for i in range(K)]

    @staticmethod
    def from_data(K, alpha, beta, D, vocab_size, cluster_doc_count, cluster_word_count, cluster_word_distribution):
        mgp = MovieGroupProcess(K, alpha, beta, n_iters=30)
        mgp.number_docs = D
        mgp.vocab_size = vocab_size
        mgp.cluster_doc_count = cluster_doc_count
        mgp.cluster_word_count = cluster_word_count
        mgp.cluster_word_distribution = cluster_word_distribution
        return mgp

    @staticmethod
    def _sample(p):
        return [i for i, entry in enumerate(multinomial(1, p)) if entry != 0][0]

    def fit(self, docs, vocab_size):
        alpha, beta, K, n_iters, V, verbose = self.alpha, self.beta, self.K, self.n_iters, vocab_size, self.verbose

        D = len(docs)
        self.number_docs = D
        self.vocab_size = vocab_size

        # unpack to easy var names
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution
        cluster_count = K
        d_z = [None for i in range(len(docs))]

        # initialize the clusters
        for i, doc in enumerate(docs):

            # choose a random  initial cluster for the doc
            z = self._sample([1.0 / K for _ in range(K)])
            d_z[i] = z
            m_z[z] += 1
            n_z[z] += len(doc)

            for word in doc:
                if word not in n_z_w[z]:
                    n_z_w[z][word] = 0
                n_z_w[z][word] += 1

        for _iter in range(n_iters):
            total_transfers = 0

            for i, doc in enumerate(docs):

                # remove the doc from it's current cluster
                z_old = d_z[i]

                m_z[z_old] -= 1
                n_z[z_old] -= len(doc)

                for word in doc:
                    n_z_w[z_old][word] -= 1

                    # compact dictionary to save space
                    if n_z_w[z_old][word] == 0:
                        del n_z_w[z_old][word]

                # draw sample from distribution to find new cluster
                p, _ = self.score(doc)
                z_new = self._sample(p)

                # transfer doc to the new cluster
                if z_new != z_old:
                    total_transfers += 1

                d_z[i] = z_new
                m_z[z_new] += 1
                n_z[z_new] += len(doc)

                for word in doc:
                    if word not in n_z_w[z_new]:
                        n_z_w[z_new][word] = 0
                    n_z_w[z_new][word] += 1

            cluster_count_new = sum([1 for v in m_z if v > 0])
            if verbose:
                print(
                    f"In stage {_iter}: transferred {total_transfers} clusters with {cluster_count_new} clusters populated")
            if total_transfers == 0 and cluster_count_new == cluster_count and _iter > 25:
                print("Converged.  Breaking out.")
                break
            cluster_count = cluster_count_new
        self.cluster_word_distribution = n_z_w
        return d_z

    def score(self, doc):
        alpha, beta, K, V, D = self.alpha, self.beta, self.K, self.vocab_size, self.number_docs
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution

        p = [0 for _ in range(K)]

        lD1 = log(D - 1 + K * alpha)
        doc_size = len(doc)
        for label in range(K):
            lN1 = log(m_z[label] + alpha)
            lN2 = 0
            lD2 = 0
            for word in doc:
                lN2 += log(n_z_w[label].get(word, 0) + beta)
            for j in range(1, doc_size + 1):
                lD2 += log(n_z[label] + V * beta + j - 1)
            p[label] = exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        pnorm = sum(p)
        pnorm = pnorm if pnorm > 0 else 1
        return [pp/pnorm for pp in p], p

    def choose_best_label(self, doc):
        p_norm, p = self.score(doc)
        return np.argmax(p), max(p_norm), max(p)


############### Download ###############


def to_excel(df, df_empty):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='clustered')
    df_empty.to_excel(writer, index=False,
                      sheet_name='unclustered')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

############### Main ###############


if __name__ == "__main__":
    main()
