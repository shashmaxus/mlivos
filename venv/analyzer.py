import pandas as pd
import numpy as np
import pickle
import time

from timeit import default_timer as timer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from common import Environment
from sharedsvc import Word_Encoder
from corpus import OpenCorpus
from postagger import POSTagger

import matplotlib.pyplot as plt
from matplotlib import rcParams

class mlAnalyzer:

    #Prepare text
    def preprocessor(self, text):
        text2 = text
        return (text2)

    #Process texts files
    def process_from_texts_file(self, aidtext, preprocessor = True):
        env = Environment()
        file_res = env.filename_results_csv()
        dfres = pd.read_csv(file_res, index_col='idstat', encoding='utf-8')
        df_texts=pd.read_csv(env.filename_texts_csv(), index_col='idtext', encoding='utf-8')
        mask = df_texts.index.isin(aidtext)
        df_texts=df_texts[mask]
        for index,row in df_texts.iterrows():
            file_txt = df_texts.at[index, 'filename']
            #Read text file
            env.debug(1, ['START file TXT:', file_txt])
            t_start = timer()
            file = open(file_txt, 'r')
            text = [file.read()]
            file.close()
            # print(text)
            idauthor = df_texts.at[index, 'idauthor']
            name = df_texts.at[index, 'name']
            s_res = self.analyze_text(text, preprocessor, index, idauthor, name, file_txt) #Analyze text, get Series
            dfres = dfres.append(s_res, ignore_index=True)
            t_end = timer()
            env.debug(1, ['END file TXT:', file_txt, 'time:', env.job_time(t_start, t_end)])
            # print(dfres.head())
        dfres = dfres.reset_index(drop=True)
        dfres.index.name = 'idstat'
        dfres.to_csv(file_res, encoding='utf-8')

    # Analyze text
    def analyze_text(self, text_to_analyze, preprocessor = True, index = 0, idauthor = 0, name = '', file_txt=''):
        env = Environment()
        enc = Word_Encoder()
        postg = POSTagger()
        corpus = OpenCorpus()
        dfgram = corpus.grammemes()
        file_authors = env.filename_authors_csv()
        authors = pd.read_csv(file_authors, index_col='idauthor', encoding='utf-8')
        t_start = timer()
        if preprocessor:
            text = self.preprocessor(text_to_analyze)
        t_end = timer()
        env.debug(1, ['Preprocessed:', 'time:', env.job_time(t_start, t_end)])
        t_start = timer()
        # Vectorize
        vectorizer = CountVectorizer()
        # analyze=vectorizer.build_analyzer()
        # print(analyze(text))
        X = vectorizer.fit_transform(text)
        #print(vectorizer.get_feature_names())
        # print(X)
        # vector = vectorizer.transform(text)
        # print(vector)
        # чтобы узнать количественное вхождение каждого слова:
        t_end = timer()
        env.debug(1, ['Vectorized:', 'time:', env.job_time(t_start, t_end)])
        t_start = timer()

        matrix_freq = np.asarray(X.sum(axis=0)).ravel()
        # final_matrix = np.array([np.array(vectorizer.get_feature_names()), matrix_freq])
        # final_matrix=list(zip(vectorizer.get_feature_names(),matrix_freq))
        # print(final_matrix)
        # words = vectorizer.vocabulary_
        # for word in words.keys():
        #    print(word,words.get(word))
        # vector = vectorizer.transform(text)
        zl = zip(vectorizer.get_feature_names(), matrix_freq)  # words, count

        data_cols = ['word', 'count', 'gram', 'gram_voc', 'gram_ml']
        data = pd.DataFrame(columns=data_cols)
        a_predict = np.array([enc.word2token('')])
        # print(a_predict)
        for feature, count in zl:
            s = pd.Series(data=[feature, count, '', '', ''], index=data_cols)
            data = data.append(s, ignore_index=True)
            a_padd = np.array([enc.word2token(feature)])
            # print(a_padd)
            # print(a_predict.shape)
            # print(a_padd.shape)
            a_predict = np.append(a_predict, a_padd, axis=0)
            # a_predict=np.concatenate((a_predict,a_padd))

        t_end = timer()
        env.debug(1, ['Ready for POS:', 'time:', env.job_time(t_start, t_end)])
        t_start = timer()

        #print(data)
        #return 0
        data = postg.pos(data)
        #print(data)
        #return(0)

        t_end = timer()
        env.debug(1, ['POS tagged:', 'time:', env.job_time(t_start, t_end)])
        t_start = timer()

        # print(data.head(1000))
        n_size = data['count'].sum()
        # print(n_size)
        grouped = data.groupby(['gram'])
        # print(grouped.groups)
        # serie_stat=grouped['gram'].count()/n_size
        serie_stat = grouped['count'].sum() / n_size
        # print(serie_stat)
        # print(serie_stat.sum())
        s_res = pd.Series([])
        s_res['idtext'] = index
        s_res['idauthor'] = idauthor
        index_author = authors.index.get_loc(idauthor)
        s_res['author'] = authors.at[index_author, 'shortname']
        s_res['name'] = name
        s_res['file'] = file_txt
        s_res['words'] = n_size
        s_res['uniq_words'] = data.shape[0]
        s_res['uniq_per_words'] = round(data.shape[0] / n_size, 4)
        for index, row in dfgram.iterrows():
            name_col = dfgram.at[index, 'name']
            try:
                s_res[name_col] = round(serie_stat[name_col], 4)
            except:
                s_res[name_col] = 0
            # print(name_col, serie_stat[name_col])
        t_end = timer()
        env.debug(1, ['Analyzed', 'time:', env.job_time(t_start, t_end)])

        return s_res

    # Return textsresults dataset
    def stat(self):
        env = Environment()
        data = pd.DataFrame()
        file_stat = env.filename_results_csv()
        try:
            data = pd.read_csv(file_stat, index_col='idstat', encoding='utf-8')
        except:
            env.debug(1, ['Failed to read stat file:', file_stat])
        else:
            env.debug(1, ['Read stat file OK:', file_stat])
        #print(data)
        return data

    #Train model
    def train(self):
        env = Environment()
        data = self.stat()
        t_start = timer()
        #print(data)
        values = data.values
        X = values[:, 7:]
        y = values[:, 1]
        y = y.astype('int')
        print(X, y)
        seed = 241
        scoring = 'accuracy'
        n_splits = 2
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        #clf = DecisionTreeClassifier(criterion='gini', random_state=seed)
        clf = GradientBoostingClassifier(n_estimators=50)
        scores = cross_val_score(clf, X, y, cv=kf)
        #clf.fit(X, y)
        print(scores)

    def get_cmap(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)


    def vizualize2d(self):
        n_components = 2
        env = Environment()
        file_res = env.filename_results_csv()
        file_authors = env.filename_authors_csv()
        data = pd.read_csv(file_res, index_col='idstat', encoding='utf-8')
        authors = pd.read_csv(file_authors, index_col='idauthor', encoding='utf-8')
        authors_v = authors.values
        values = data.values
        X = values[:, 8:]
        y = values[:, 1]
        #print(data)
        #print(X, y)
        pca=PCA(n_components=2)
        #pca = TSNE(n_components=2)
        X_new = pca.fit_transform(X, y)
        print('PCA ratio 2 components', pca.explained_variance_ratio_)
        #print('components', pca.components_)
        #print(X_new)
        tdf = pd.DataFrame(data=X_new, columns=['PC1', 'PC2'])
        finalDf = pd.concat([tdf, data[['idauthor','name']]], axis = 1)
        print('dataframe ', finalDf)

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Tahoma']
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Component 1. Вклад '+str(round(pca.explained_variance_ratio_[0],2)), fontsize=12)
        ax.set_ylabel('Component 2. Вклад '+str(round(pca.explained_variance_ratio_[1],2)), fontsize=12)
        ax.set_title('2 component PCA. Точность '+str(round(sum(float(i) for i in pca.explained_variance_ratio_),2)), fontsize=12)
        targets = authors.index.values
        legends = authors_v[:, 0]
        cmap = self.get_cmap(len(targets))
        #print(targets)
        #colors = ['r', 'g', 'b']
        colors = "bgcmykw" #without r
        for target in targets:
            #print(cmap(target))
            indicesToKeep = finalDf['idauthor'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
                       , finalDf.loc[indicesToKeep, 'PC2']
                       , c = colors [target]
                       , s=50)
        for index, row in finalDf.iterrows():
            ax.annotate(finalDf.at[index, 'name'], (finalDf.at[index, 'PC1'], finalDf.at[index, 'PC2']), fontsize=8)
        ax.legend(legends)
        ax.grid()
        plt.show()