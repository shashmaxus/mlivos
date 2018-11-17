import pandas as pd
import numpy as np
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

from common import Environment
from sharedsvc import Word_Encoder
from corpus import OpenCorpus
from postagger import POSTagger

import matplotlib.pyplot as plt

class mlAnalyzer:

    #Prepare text
    def preprocessor(self, text):
        text2 = text
        return (text2)

    #Analyze text file
    def analyze_text(self, aidtext, preprocessor = True):
        env = Environment()
        enc = Word_Encoder()
        corpus = OpenCorpus()
        postg = POSTagger()
        #file_model = env.filename_model_tree()
        dfgram = corpus.grammemes()
        file_res = env.filename_results_csv()
        dfres = pd.read_csv(file_res, index_col='idstat', encoding='utf-8')
        file_authors = env.filename_authors_csv()
        authors = pd.read_csv(file_authors, index_col='idauthor', encoding='utf-8')

        df_texts=pd.read_csv(env.filename_texts_csv(), index_col='idtext', encoding='utf-8')
        mask = df_texts.index.isin(aidtext)
        df_texts=df_texts[mask]
        for index,row in df_texts.iterrows():
            file_txt = df_texts.at[index, 'filename']
            #Read text file
            env.debug(1, ['START file TXT:', file_txt])
            file = open(file_txt, 'r')
            text = [file.read()]
            file.close()
            # print(text)

            #Vectorize
            vectorizer = CountVectorizer()
            # analyze=vectorizer.build_analyzer()
            # print(analyze(text))
            X = vectorizer.fit_transform(text)
            # print(vectorizer.get_feature_names())
            # print(X)
            # vector = vectorizer.transform(text)
            # print(vector)

            # чтобы узнать количественное вхождение каждого слова:
            matrix_freq = np.asarray(X.sum(axis=0)).ravel()
            # final_matrix = np.array([np.array(vectorizer.get_feature_names()), matrix_freq])
            # final_matrix=list(zip(vectorizer.get_feature_names(),matrix_freq))
            # print(final_matrix)
            # words = vectorizer.vocabulary_
            # for word in words.keys():
            #    print(word,words.get(word))
            # vector = vectorizer.transform(text)
            zl = zip(vectorizer.get_feature_names(), matrix_freq) #words, count

            data_cols = ['word', 'count','gram','gram_voc','gram_ml']
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
            data = postg.pos(data)
            #print(data.head(1000))

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
            s_res['idauthor'] = df_texts.at[index, 'idauthor']
            index_author =  authors.index.get_loc(df_texts.at[index, 'idauthor'])
            s_res['author'] = authors.at [index_author, 'shortname']
            s_res['name'] = df_texts.at[index, 'name']
            s_res['file'] = file_txt
            s_res['words'] = n_size
            s_res['uniq_words'] = data.shape[0]
            s_res['uniq_per_words'] = round(data.shape[0] / n_size,4)
            for index, row in dfgram.iterrows():
                name_col = dfgram.at[index, 'name']
                try:
                    s_res[name_col] = round(serie_stat[name_col],4)
                except:
                    s_res[name_col] = 0
                # print(name_col, serie_stat[name_col])
            # print(s_res)
            dfres = dfres.append(s_res, ignore_index=True)
            env.debug(1, ['END file TXT:', file_txt])
            # print(dfres.head())
        dfres = dfres.reset_index(drop=True)
        dfres.index.name = 'idstat'
        dfres.to_csv(file_res, encoding='utf-8')

            # print(grouped['gram'].agg({'count' : count / n_size}))
            # print(row)
            # print(list(zip(a_predict[:, 0], predictions)))

            # predictions = clf.predict(a_smoke[:, 6:12])
            # a_smoke = np.array(
            #    [enc.word2token('съеште'), enc.word2token('ещё'), enc.word2token('этих'),
            #     enc.word2token('мягких'), enc.word2token('французских'),enc.word2token('булок')])
            # print(a_smoke)
            # predictions = clf.predict(a_smoke[:, 6:12])
            # print(list(zip(a_smoke[:, 0], predictions)))

    def get_cmap(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)


    def analyze_results(self):
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
        X_new = pca.fit_transform(X, y)
        print('ratio', pca.explained_variance_ratio_)
        #print('components', pca.components_)
        #print(X_new)
        tdf = pd.DataFrame(data=X_new, columns=['PC1', 'PC2'])
        finalDf = pd.concat([tdf, data[['idauthor']]], axis = 1)
        print(finalDf)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        targets = authors.index.values
        legends = authors_v[:, 0]
        cmap = self.get_cmap(len(targets))
        #print(targets)
        #colors = ['r', 'g', 'b']
        colors = "bgrcmykw"
        for target in targets:
            #print(cmap(target))
            indicesToKeep = finalDf['idauthor'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
                       , finalDf.loc[indicesToKeep, 'PC2']
                       , c = colors [target]
                       , s=50)
        ax.legend(legends)
        ax.grid()
        plt.show()