import pandas as pd
import numpy as np
import pickle
import time
import string, re
import codecs

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer

from timeit import default_timer as timer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold

#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


from common import Environment
from sharedsvc import Word_Encoder
from corpus import OpenCorpus
from postagger import POSTagger

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.style

class mlAnalyzer:

    @staticmethod
    def punctuation():
        return string.punctuation + '\u2014\u2013\u2012\u2010\u2212' + '«»‹›‘’“”„`'

    @staticmethod
    def word_tokenizers_custom():
        #return re.compile(r"([^\w_\u2019\u2010\u002F\u0027-]|[+])")
        return (r'\w+')

    #Prepare text
    def preprocessor(self, text):
        env = Environment()
        t_start = timer()
        text2 = text.lower()
        env.debug(1, ['Analyzer','preprocessor','START Preprocessing:'])
        tokenizer = RegexpTokenizer(self.word_tokenizers_custom())
        tokens_words = tokenizer.tokenize(text2) #Слова текста
        tokens_sent = sent_tokenize(text2) #Предложения - пока не используются в нашем проекте

        n_words_count = len(tokens_words) #Количество слов в тексте
        n_sent_count = len(tokens_sent) #Количество предложений в тексте

        #Средняя длина предложения в словах
        sent_len = np.zeros(n_sent_count)
        for i in range (0, n_sent_count):
            sent_l1 = tokenizer.tokenize(tokens_sent[i])
            sent_len[i] = len(sent_l1)
        n_sent_len_mean = sent_len.mean() #средняя длина предложения в словах
        t_end = timer()
        env.debug(1, ['Preprocessed:', 'time:', env.job_time(t_start, t_end)])
        return (n_sent_count, n_sent_len_mean, n_words_count, tokens_words)

    #Process texts files
    def process_from_texts_file(self, aidtext):
        env = Environment()
        file_res = env.filename_results_csv()
        dfres = pd.read_csv(file_res, index_col='idstat', encoding='utf-8') #Файл для записи статистических результатов
        #dfres = env.downcast_dtypes(dfres)
        df_texts = pd.read_csv(env.filename_texts_csv(), index_col='idtext', encoding='utf-8') #Реестр текстов
        mask = df_texts.index.isin(aidtext)
        df_texts = df_texts[mask]
        for index, row in df_texts.iterrows(): #Для каждого текста, который надо обработать
            file_txt = df_texts.at[index, 'filename']
            #Read text file
            env.debug(1, ['START file TXT:', file_txt])
            t_start = timer()
            #file = open(file_txt, 'r')
            file = codecs.open(file_txt, "r", "utf_8_sig")
            text = file.read().strip()
            file.close()
            # print(text)
            #Автор в обучающей выборке указанг
            idauthor = df_texts.at[index, 'idauthor'] #Автор
            name = df_texts.at[index, 'name'] #Название

            #Собственно обработка текста
            df_add = self.analyze_text(dfres.columns, text, index, idauthor, name, file_txt) #Analyze text, get Series
            df_add.reset_index(drop = True, inplace = True)
            dfres = dfres.append(df_add, ignore_index=True) #Добавляем к файлу результатов
            dfres.reset_index(drop = True, inplace = True)
            dfres.index.name = 'idstat'
            #print(dfres)
            #return 0
            t_end = timer()
            env.debug(1, ['END file TXT:', file_txt, 'time:', env.job_time(t_start, t_end)])
            # print(dfres.head())
        #Сохраняем результат на диск
        #dfres = dfres.reset_index(drop=True)
        int_cols = ['idtext', 'idchunk', 'idauthor', 'words_all', 'words_chunk', 'sentences_all', 'words_uniq']
        for col in int_cols:
            dfres[col] = dfres[col].astype(int)
        #dfres = env.downcast_dtypes(dfres)
        dfres.to_csv(file_res, encoding='utf-8')



    # Analyze text
    def analyze_text(self, columns, text_to_analyze, index = 0, idauthor = 0, name = '', file_txt=''):
        env = Environment()
        t_start = timer()
        env.debug(1, ['Analyzer', 'analyze_text', 'START file TXT: %s' % file_txt])
        enc = Word_Encoder()
        postg = POSTagger()
        corpus = OpenCorpus()
        dfgram = corpus.grammemes()
        file_authors = env.filename_authors_csv()
        #Информация об авторах
        authors = pd.read_csv(file_authors, index_col='idauthor', encoding='utf-8')

        dfres = pd.DataFrame()

        #Preprocessing: выполнить прдварительную обработку текста
        n_sent_count, n_sent_len_mean, n_words_count, tokens_words = self.preprocessor(text_to_analyze)
        #на выходе получили число предложений, число слов в тексте, массив со словами
        env.debug(1, ['Analyzer', 'analyze_text', 'get %s words in %s sentences' % (n_words_count, n_sent_count)])

        #Если документ большой, разделяем его на несколько частей (chunks) и считаем
        #статистику для каждого в отдельности.
        # Это нам позволит имея небольшое число объёмных документов корректно обучить модель
        n_chunks = n_words_count // env.analyzer_max_words + 1 #на сколько частей будем делить
        #print(n_chunks)
        gen_chunks =  env.chunks(tokens_words, n_chunks) #разделили
        a_text_corp = []
        id_chunk = 0
        for a_text_words in gen_chunks:
            #a_text_corp.append(' '.join(map(str,a_text_words)))
            # Vectorize - каждую часть индивидуально
            vectorizer = CountVectorizer()
            #Преобразуем все слова в матрицу из одной строки (0) и множества колонок, где каждому слову соотвествует
            # колонка, а количество вхождений слова в документе - значение в этой колонке
            X = vectorizer.fit_transform([' '.join(map(str,a_text_words))])
            n_words_chunk = X.sum() #Сколько всего слов в документе, который обрабатываем
            env.debug(1, ['Analyzer', 'analyze_text', 'START process chunk %s/%s with %s words' % (id_chunk, n_chunks, n_words_chunk)])
            word_freq = np.asarray(X.sum(axis=0)).ravel() #для каждого слова его суммарное число (т.к. у нас одна строка == числу в ней)
            #print(vectorizer.get_feature_names())
            #print(X)
            zl = zip(vectorizer.get_feature_names(), word_freq)  # words, count
            #print(list(zl))

            t_start = timer()

            data_cols = ['gram', 'gram_voc', 'gram_ml']
            data = pd.DataFrame(list(zl),columns=['word', 'count'])
            for col in data_cols:
                data[col] = ''
            #print(data)

            #a_predict = np.array([enc.word2token('')])
            #for s_word in vectorizer.get_feature_names():
                #a_enc_word = enc.word2token(s_word)
                #a_predict.append(a_enc_word)
                #a_enc_word = np.array([enc.word2token(s_word)])
                #a_predict = np.append(a_predict, a_enc_word, axis=0)
            #print(a_predict)
            t_end = timer()
            env.debug(1, ['Ready for POS:', 'time:', env.job_time(t_start, t_end)])

            t_start = timer()
            data = postg.pos(data)
            #print(data)
            t_end = timer()
            env.debug(1, ['POS tagged:', 'time:', env.job_time(t_start, t_end)])

            t_start = timer()
            grouped = data.sort_values('gram').groupby(['gram']).agg({'count' : ['sum']})
            grouped.columns = ['n_POS']
            grouped.reset_index(inplace=True)
            grouped['f_POS'] = grouped['n_POS'] / n_words_chunk
            #grouped.drop(columns=['n_POS'], inplace=True)
            #print(grouped)
            #print(grouped.set_index('gram').T)
            grouped = pd.merge(dfgram, grouped, left_on='name', right_on='gram', how='left').drop(
                columns=['alias', 'description', 'name', 'n_POS']).fillna(0).set_index('gram').T
            #grouped = pd.merge(dfgram, grouped, left_on='name', right_on='gram', how='left').fillna(0).set_index('gram')
            #print(grouped)
            #print(grouped.values.ravel())
            index_author = authors.index.get_loc(idauthor)
            s_chunk = pd.Series({'idtext': index,
                                     'idchunk' : id_chunk,
                                     'idauthor' : idauthor,
                                     'author' : authors.at[index_author, 'shortname'],
                                     'name' : name,
                                     'file' : file_txt,
                                     'words_all' : np.int64(n_words_count),
                                     'words_chunk' : n_words_chunk,
                                     'sentences_all' : np.int64(n_sent_count),
                                     'sentence_mean': n_sent_len_mean,
                                     'words_uniq' : data.shape[0],
                                     'uniq_per_words' : round(data.shape[0] / n_words_chunk, 4)
                                     })
            s_chunk = pd.concat([s_chunk, pd.Series(grouped.values.ravel())], ignore_index=True)

            #print(s_chunk)
            #print(grouped)
            t_end = timer()
            env.debug(1, ['Analyzed', 'time:', env.job_time(t_start, t_end)])
            dfres = dfres.append(s_chunk, ignore_index=True)
            #dfres = env.downcast_dtypes(dfres)
            id_chunk = id_chunk + 1
        print(dfres)
        dfres.columns = columns
        return dfres

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

    #Prepare data for machine learning algoritms
    def model_prepare_data(self, df, mode = 'train'):
        env = Environment()
        data = df.copy()
        data.drop(columns=['file', 'idchunk','predict'], inplace=True)

        columns = data.columns

        #idstat,idtext,idchunk,idauthor,author,name,file,words_all,words_chunk,sentences_all,sentence_mean,words_uniq,uniq_per_words,NOUN,ADJF,ADJS,COMP,VERB,INFN,PRTF,PRTS,GRND,NUMR,ADVB,NPRO,PRED,PREP,CONJ,PRCL,INTJ

        columns2drop = ['idtext', 'idauthor', 'author', 'name',
                        'words_all', 'words_chunk', 'sentences_all', 'words_uniq']

        #New features
        #Создадим новые статистические поля для помощи нашей модели
        data['words_uniq_per_sentense'] = data['words_uniq'] / data['sentences_all'] #кол-во уникальных слов/ кол-во предложений
        #data['words_uniq_3k'] = data['words_uniq'] / 3000  # кол-во уникальных слов на 3 тыс. слов
        #data['words_uniq_10k'] = data['words_uniq'] / 10000 #кол-во уникальных слов на 10 тыс. слов

        y = None
        if mode == 'train':
            y = data['idauthor']
        X = data.drop(columns = columns2drop)

        #Add PCA features
        n_components = 2
        pca_cols2drop = ['sentence_mean', 'uniq_per_words', 'words_uniq_per_sentense']
        if mode == 'train': #формируем матрицу признаков
            pca_pos = PCA(n_components = n_components)
            X_new = pca_pos.fit_transform(X.drop(columns = pca_cols2drop), y)
            print('PCA ratio %s components quality: %s' % (n_components, round(np.sum(pca_pos.explained_variance_ratio_),4)), pca_pos.explained_variance_ratio_)
            pickle.dump(pca_pos, open(env.filename_model_texts_pca(), 'wb'))
        if mode == 'test': #Переводим признаки в пространство признаков на основе ранее созданной матрицы
            pca_pos = pickle.load(open(env.filename_model_texts_pca(), 'rb'))
            X_new = pca_pos.transform(X.drop(columns = pca_cols2drop))
        for i in range (0, n_components):
            X['pca_%s' % i] = X_new[:, i]
        return y, X

    #Train model
    def model_train(self):
        env = Environment()
        data = self.stat()
        t_start = timer()
        y, X = self.model_prepare_data(data)

        seed = 241
        scoring = 'accuracy'
        n_splits = 4
        frac_test_size = 0.25

        #Cross-validation
        kf = KFold(n_splits = n_splits, shuffle = True, random_state = seed)
        #clf = DecisionTreeClassifier(criterion='gini', random_state=seed)
        #clf = GradientBoostingClassifier(n_estimators=50)
        model = xgb.XGBClassifier(n_estimators = 200, max_depth = 8, colsample = 1, subsample = 1, seed = seed)
        cv_scores = cross_val_score(model, X, y, cv = kf)

        #eval
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = frac_test_size, random_state = seed)
        eval_set = [(X_train, y_train), (X_test, y_test)]
        #print(eval_set)
        f_eval = 'merror'
        # f_eval = 'mlogloss'
        model.fit(X_train, y_train, eval_metric=f_eval, eval_set = eval_set, verbose=False, early_stopping_rounds = 10)
        ev_scores = model.evals_result()

        cv_mean = np.array(cv_scores.mean())
        #ev_mean = np.array(ev_scores['validation_0']['mlogloss']).mean()
        ev_mean = np.array(ev_scores['validation_0'][f_eval]).mean()

        #Посмотрим важность признаков в модели
        #print(model.feature_importances_)
        xgb.plot_importance(model)
        #plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
        plt.show()

        #Обучаем модель на всех данных
        model.fit(X, y, verbose=False)
        #Сохраняем модель на диск
        pickle.dump(model, open(env.filename_model_texts(), 'wb'))

        #print('CV', cv_scores, 'EV', ev_scores)
        print('Cross-validation: mean', cv_mean, 'eval_set mean', ev_mean)
        return model

    def model_predict(self, df, b_retrain = False):
        env = Environment()
        y, X = self.model_prepare_data(df, mode='test')
        if b_retrain:
            model = self.model_train() #Если хотим для кажжого теста вновь тренировать модель
        else:
            #Загружаем ранее тренированную модель с диска
            model = pickle.load(open(env.filename_model_texts(), 'rb'))
        #Предсказываем
        y = model.predict(X)
        return y

    #Predict
    def predict(self, aidtext, b_makestat = False):
        env = Environment()

        # Открываем файл со статистикой по тестовым текстам
        df_stat = pd.read_csv(env.filename_stat_test_csv(), index_col='idstat', encoding='utf-8')  # Статистика по тстовым текстам

        df_texts = pd.read_csv(env.filename_predict_csv(), index_col='idtext', encoding='utf-8')  # Реестр текстов
        mask = df_texts.index.isin(aidtext)
        df_texts = df_texts[mask]

        columns = ['idtext', 'idchunk', 'idauthor', 'author', 'name', 'file', \
                  'words_all', 'words_chunk', 'sentences_all', 'sentence_mean', \
                  'words_uniq','uniq_per_words', \
                  'NOUN','ADJF','ADJS','COMP','VERB','INFN','PRTF','PRTS','GRND','NUMR',\
                  'ADVB','NPRO','PRED','PREP','CONJ','PRCL','INTJ']
        y_result = []

        #Если необходимо подготовить статистику по тестовым текстам
        if b_makestat:
            for index, row in df_texts.iterrows():  # Для каждого текста, который надо обработать
                file_txt = df_texts.at[index, 'filename']
                # Read text file
                env.debug(1, ['Analyzer','predict','START file TXT:', file_txt])
                t_start = timer()
                file = codecs.open(file_txt, "r", "utf_8_sig")
                text = file.read().strip()
                file.close()
                # Автор в тестовой выборке вообще говоря нет
                idauthor = df_texts.at[index, 'idauthor']  # Автор
                #idauthor = 0
                name = df_texts.at[index, 'name']  # Название

                # Собственно обработка текста
                df_add = self.analyze_text(columns, text, index, idauthor, name,
                                           file_txt)  # Analyze text, get Series
                #print(df_add)
                df_add.reset_index(drop = True, inplace = True)
                df_stat = df_stat.append(df_add, ignore_index=True) #Добавляем к файлу результатов
                df_stat.reset_index(drop = True, inplace = True)
                df_stat.index.name = 'idstat'
                t_end = timer()
                env.debug(1, ['END file TXT:', file_txt, 'time:', env.job_time(t_start, t_end)])
            #df_stat теперь содержит информацию о всех тестовых текстах, которые хотели обработать
            #Указываем верный тип для целочисленных колонок
            int_cols = ['idtext', 'idchunk', 'idauthor', 'words_all', 'words_chunk', 'sentences_all', 'words_uniq']
            for col in int_cols:
                df_stat[col] = df_stat[col].astype(int)
            # Сохраняем результат на диск
            df_stat.to_csv(env.filename_stat_test_csv(), encoding='utf-8')
        #Статистика готова

        # Открываем файл со статистикой по тестовым текстам
        df_stat = pd.read_csv(env.filename_stat_test_csv(), index_col='idstat', encoding='utf-8')  # Статистика по тстовым текстам
        #mask = df_stat.index.isin(aidtext)
        #df_stat2predict = df_stat[mask]
        #Предсказываем авторов
        y_res = self.model_predict(df_stat.loc[aidtext])
        #print(y_res)
        df_stat.loc[aidtext, 'predict'] = y_res
        #print(df_stat)
        #y_result.append(y_res[0])
        #Сохраняем измененный файл с предсказаниями
        df_stat.to_csv(env.filename_stat_test_csv(), encoding='utf-8')
        return y_res #Возвращаем предсказания

    def get_texts_stat(self, mode = 'train'):
        # Готовим данные
        env = Environment()
        if mode == 'train':
            file_res = env.filename_results_csv()
        if mode == 'test':
            file_res = env.filename_stat_test_csv()
        authors = pd.read_csv(env.filename_authors_csv(), index_col='idauthor', encoding='utf-8')

        data = pd.read_csv(file_res, index_col='idstat', encoding='utf-8')
        data.drop(columns=['file', 'idchunk'], inplace=True)
        columns = data.columns

        group = data.groupby(['idtext', 'idauthor', 'author', 'name'])
        group = group.agg({'words_all': ['mean'],
                           'words_chunk': ['mean'],
                           'sentences_all': ['mean'],
                           'sentence_mean': ['mean'],
                           'words_uniq': ['mean'],
                           'uniq_per_words': ['mean'],
                           'NOUN': ['mean'],
                           'ADJF': ['mean'],
                           'ADJS': ['mean'],
                           'COMP': ['mean'],
                           'VERB': ['mean'],
                           'INFN': ['mean'],
                           'PRTF': ['mean'],
                           'PRTS': ['mean'],
                           'GRND': ['mean'],
                           'NUMR': ['mean'],
                           'ADVB': ['mean'],
                           'NPRO': ['mean'],
                           'PRED': ['mean'],
                           'PREP': ['mean'],
                           'CONJ': ['mean'],
                           'PRCL': ['mean'],
                           'INTJ': ['mean'],
                           'predict' : ['sum']})
        group.columns = columns[4:]
        group.reset_index(inplace=True)
        data = pd.merge(group, authors, on='idauthor', how='left', suffixes=('', '_author'))
        return data

    def vizualize2d(self, mode='train'):
        n_components = 2
        env = Environment()
        if mode == 'train':
            file_res = env.filename_results_csv()
        if mode == 'test':
            file_res = filename_stat_test_csv()
        file_authors = env.filename_authors_csv()
        data = pd.read_csv(file_res, index_col='idstat', encoding='utf-8')

        data.drop(columns=['file', 'idchunk'], inplace=True)
        columns = data.columns

        columns2drop = ['idtext', 'idauthor', 'author', 'name',
         'words_all', 'words_chunk', 'sentences_all', 'words_uniq', 'shortname']

        group = data.groupby(['idtext','idauthor', 'author', 'name'])
        group = group.agg({'words_all': ['mean'],
                           'words_chunk': ['mean'],
                           'sentences_all' : ['mean'],
                            'sentence_mean' : ['mean'],
                            'words_uniq' : ['mean'],
                            'uniq_per_words' : ['mean'],
                            'NOUN' : ['mean'],
                            'ADJF': ['mean'],
                            'ADJS': ['mean'],
                            'COMP' : ['mean'],
                            'VERB' : ['mean'],
                            'INFN' : ['mean'],
                            'PRTF' : ['mean'],
                            'PRTS' : ['mean'],
                            'GRND' : ['mean'],
                            'NUMR' : ['mean'],
                            'ADVB' : ['mean'],
                            'NPRO' : ['mean'],
                            'PRED' : ['mean'],
                            'PREP' : ['mean'],
                            'CONJ' : ['mean'],
                            'PRCL' : ['mean'],
                            'INTJ' : ['mean']})

        #print(data)
        #print(group, group.info())
        #print(group.columns)
        #print(columns)

        group.columns = columns[4:]
        group.reset_index(inplace=True)
        #print(group)

        authors = pd.read_csv(file_authors, index_col='idauthor', encoding='utf-8')
        authors_v = authors.values


        data = pd.merge(group, authors, on='idauthor', how='left', suffixes=('','_')).drop(columns=['name_'])
        #print(data.columns)

        y = data['idauthor'].values
        X = data.drop(columns=columns2drop).values
        #X = data.values

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
        #print('dataframe ', finalDf)

        mpl.style.use('default')

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Tahoma']

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Component 1. Вклад '+str(round(pca.explained_variance_ratio_[0],2)), fontsize=12)
        ax.set_ylabel('Component 2. Вклад '+str(round(pca.explained_variance_ratio_[1],2)), fontsize=12)
        ax.set_title('2 component PCA. Точность '+str(round(sum(float(i) for i in pca.explained_variance_ratio_),2)), fontsize=12)
        targets = authors.index.values
        legends = authors_v[:, 0]
        #print(targets)
        #colors = ['r', 'g', 'b']
        #colors = "bgcmykw" #without r
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                      '#bcbd22', '#17becf']
        for target in targets:
            indicesToKeep = finalDf['idauthor'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
                       , finalDf.loc[indicesToKeep, 'PC2']
                       , c = colors [target]
                       , s=50)
        for index, row in finalDf.iterrows():
            ax.annotate(finalDf.at[index, 'name'],
                        xy= (finalDf.at[index, 'PC1'], finalDf.at[index, 'PC2']),
#                        xytext=(0.05, 0.05),
                        fontsize=8)
        ax.legend(legends)
        ax.grid()
        plt.show()