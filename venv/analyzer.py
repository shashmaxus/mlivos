import pandas as pd
import numpy as np
import pickle
import time
import string, re
import codecs
import math

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
from sklearn.metrics import r2_score

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

    #Process texts files
    def process_from_texts_file(self, aidtext, mode = 'process', max_words = 0):
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
            columns = dfres.columns
            if mode == 'process': #если необходимо собрать информацию о тексте и записать её в results
                #Собственно обработка текста
                df_add = self.analyze_text(columns, text, index, idauthor, name, file_txt, max_words) #Analyze text, get Series
                df_add.reset_index(drop = True, inplace = True)
                dfres = dfres.append(df_add, ignore_index=True) #Добавляем к файлу результатов
                dfres.reset_index(drop = True, inplace = True)
                dfres.index.name = 'idstat'
                #print(dfres)
                #return 0

            if mode == 'chunk_size': # если необходимо определить размер chunk
                n_chunk_size = self.validate_chunk_size(columns, text, index, idauthor, name, file_txt)
            t_end = timer()
            env.debug(1, ['END file TXT:', file_txt, 'time:', env.job_time(t_start, t_end)])
            # print(dfres.head())
        #Сохраняем результат на диск
        if mode == 'process':
            #dfres = dfres.reset_index(drop=True)
            int_cols = ['idtext', 'idchunk', 'idauthor',
                        'sentences_text', 'words_text',
                        'sentences_chunk', 'words_chunk',  'words_uniq_chunk']
            for col in int_cols:
                dfres[col] = dfres[col].astype(int)
            #dfres = env.downcast_dtypes(dfres)
            dfres.to_csv(file_res, encoding='utf-8')

    # Prepare text
    def preprocessor(self, text, max_words = 0):
        env = Environment()
        t_start = timer()
        text2 = text.lower()
        env.debug(1, ['Analyzer', 'preprocessor', 'START Preprocessing:'])
        tokenizer = RegexpTokenizer(self.word_tokenizers_custom())
        tokens_words = tokenizer.tokenize(text2)  # Слова текста
        tokens_sent = sent_tokenize(text2)  # Предложения - пока не используются в нашем проекте

        n_words_count = len(tokens_words)  # Количество слов в тексте
        n_sent_count = len(tokens_sent)  # Количество предложений в тексте
        n_sent_len_mean = n_words_count / n_sent_count  # Средняя длина предложения в словах

        #Делим текст на части - chunks
        awords = [] #Массив
        # Если документ большой, разделяем его на несколько частей (chunks) и считаем
        # статистику для каждого в отдельности.
        # Это нам позволит имея небольшое число объёмных документов корректно обучить модель
        if (max_words > 0):
            n_sent_chunk = int(max_words // n_sent_len_mean) #Сколько предложение в 1 chunks содержащее max_words

            print('n_sent_chunk', n_sent_chunk)
            #подбираем, чтобы текст был разделен равномерно
            i_chunks = 1
            tmp_sent_chunk = n_sent_count
            while tmp_sent_chunk > n_sent_chunk:
                i_chunks = i_chunks + 1
                tmp_sent_chunk = int (math.ceil(n_sent_count // i_chunks) + (n_sent_count % i_chunks))

            n = 0
            n_sent_chunk = tmp_sent_chunk #итоговое значение сколько предложений пойдет в chunk
            print('tmp_sent_chunk', tmp_sent_chunk)

            while n < n_sent_count:
                #print(n, n_sent_chunk)
                asents = tokens_sent[n : n + n_sent_chunk] #Предложения от n до n+chunk
                #print(asents)
                a_sent_words = [] #слова текущей группы предложений
                for sent in asents:
                    words = tokenizer.tokenize(sent)
                    a_sent_words.extend(words)
                #print(a_sent_words)
                awords.append([n_sent_count, n_words_count, len(a_sent_words)/len(asents), len(asents), len(a_sent_words), a_sent_words])
                n = n + n_sent_chunk
        else:
            awords.append([n_sent_count, n_words_count, n_sent_len_mean, len(tokens_sent), len(tokens_words), tokens_words])
        #print(awords)
        t_end = timer()
        env.debug(1, ['Preprocessed:', 'time:', env.job_time(t_start, t_end)])
        return awords #Массив со словами и статистикой

    # Analyze text
    def analyze_text(self, columns, text_to_analyze, index = 0, idauthor = 0, name = '', file_txt='', max_words = 0):
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

        dfres = pd.DataFrame() #Пустой dataframe для сохранения результат

        #Preprocessing: выполнить прдварительную обработку текста
        #max_words = 6000
        achunks = self.preprocessor(text_to_analyze, max_words)
        #print(achunks)
        n_chunks = len(achunks)  # на сколько частей разделён текст

        #на выходе получили массив, каждывй элемент которого содеоржит число предложений, число слов в тексте, массив со словами
        env.debug(1, ['Analyzer', 'analyze_text', '%s sentences %s words in %s chunks' % (achunks[0][0], achunks[0][1], n_chunks)])
        #print(n_chunks)
        a_text_corp = []
        id_chunk = 0
        for  chunk in achunks:
            t_start = timer() #prepare data
            n_sent_all = chunk[0]
            n_words_all = chunk[1]
            n_sent_len_mean = chunk[2]
            n_sent_chunk = chunk[3]
            n_words_chunk = chunk[4]
            a_text_words = chunk[5]
            #print(n_sent_all, n_words_all, n_sent_len_mean, n_sent_chunk, n_words_chunk, a_text_words)
            #print(len(a_text_words))

            # Vectorize - к каждой части относимся как к индивидуальному тексту
            vectorizer = CountVectorizer(encoding='utf-8', token_pattern=r"(?u)\b\w+\b")
            #Преобразуем все слова в матрицу из одной строки (0) и множества колонок, где каждому слову соотвествует
            # колонка, а количество вхождений слова в документе - значение в этой колонке
            #print([' '.join(map(str,a_text_words))])
            X = vectorizer.fit_transform([' '.join(map(str,a_text_words))])
            #print(X)
            n_words_chunk_check = X.sum() #Сколько всего слов в документе, который обрабатываем
            #print(n_words_chunk, n_words_chunk_check)
            #print(vectorizer.get_stop_words())

            env.debug(1, ['Analyzer', 'analyze_text', 'START process chunk %s/%s with %s words' % (id_chunk, n_chunks-1, n_words_chunk)])
            word_freq = np.asarray(X.sum(axis=0)).ravel() #для каждого слова его суммарное число (т.к. у нас одна строка == числу в ней)
            #print(vectorizer.get_feature_names())
            #print(X)
            zl = zip(vectorizer.get_feature_names(), word_freq)  # words, count
            #print(list(zl))

            data_cols = ['gram', 'gram_voc', 'gram_ml']
            data = pd.DataFrame(list(zl),columns=['word', 'count'])
            for col in data_cols:
                data[col] = ''
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
            n_uniq_words = data.shape[0]
            s_chunk = pd.Series({'idtext': index,
                                 'idchunk' : id_chunk,
                                 'idauthor' : idauthor,
                                 'author' : authors.at[index_author, 'shortname'],
                                 'name' : name,
                                 'file' : file_txt,
                                 'sentences_text': np.int64(n_sent_all),
                                 'words_text' : np.int64(n_words_all),
                                 'sentence_mean': n_sent_len_mean,
                                 'sentences_chunk': np.int64(n_sent_chunk),
                                 'words_chunk' : np.int64(n_words_chunk),
                                 'words_uniq_chunk' : np.int64(n_uniq_words),
                                 'uniq_per_sent_chunk': round(n_uniq_words / n_sent_chunk, 4),
                                 'uniq_per_words_chunk' : round(n_uniq_words / n_words_chunk, 4)
                                     })
            s_chunk = pd.concat([s_chunk, pd.Series(grouped.values.ravel())], ignore_index=True)
            s_chunk = pd.concat([s_chunk, pd.Series([np.nan])], ignore_index=True)
            #print(s_chunk)
            #print(grouped)
            t_end = timer()
            env.debug(1, ['Analyzed', 'time:', env.job_time(t_start, t_end)])
            dfres = dfres.append(s_chunk, ignore_index=True)
            #dfres = env.downcast_dtypes(dfres)
            id_chunk = id_chunk + 1
        print(dfres)
        print(columns)
        dfres.columns = columns
        return dfres

    #Определяем оптимальный размер chunk
    def validate_chunk_size(self, columns, text, index, idauthor, name, file_txt):
        #a_chunk_sizes = [1000, 3000, 5000, 7500, 10000, 15000]
        a_chunk_sizes = [100, 500, 1000, 3000, 5000, 6000, 7500, 9000, 10000, 15000, 20000]
        #a_chunk_sizes = [20000]
        cols2compare = ['NOUN', 'ADJF', 'ADJS', 'COMP', 'VERB', 'INFN', 'PRTF', 'PRTS',  #'uniq_per_sent_chunk','uniq_per_words_chunk',
                        'GRND', 'NUMR', 'ADVB', 'NPRO', 'PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ']
        dfall = self.analyze_text(columns, text, index, idauthor, name, file_txt, max_words = 0)
        X_orig = np.array(dfall[cols2compare].values.ravel())
        #print(dfall)
        #print(X_orig)
        a_chunk_stat = []
        for chunk in a_chunk_sizes:
            dfchunk = self.analyze_text(columns, text, index, idauthor, name, file_txt, max_words = chunk)
            scores = []
            for index, row in dfchunk.iterrows():
                k_res = np.array(row[cols2compare].values.ravel())
                #print(X_orig, k_res)
                #print(np.corrcoef(X_orig, k_res))
                scores.append(r2_score(X_orig, k_res))
            #print(dfchunk.describe().loc['mean'][cols2compare])
            a_chunk_stat.append([chunk, dfchunk.shape[0], np.mean(scores)])
            #print(dfchunk)
        print(a_chunk_stat)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Кол-во слов')
        ax.set_ylabel('Соотвествие статистике всего текста')
        ax.set_title('Текст: %s' % name)
        #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        #          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        #          '#bcbd22', '#17becf']
        colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
                  "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
                  "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
        i = 0
        for stat in a_chunk_stat:
            ax.scatter(x = stat[0],
                       y = stat[2],
                       c = colors[i],
                       s = 50)
            i = i+1;
        ax.grid()

        plt.show()

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
                        'sentences_text', 'words_text', 'sentences_chunk', 'words_chunk', 'words_uniq_chunk']

        #New features
        #Создадим новые статистические поля для помощи нашей модели
        #data['words_uniq_per_sentense'] = data['words_uniq'] / data['sentences_all'] #кол-во уникальных слов/ кол-во предложений
        #data['words_uniq_3k'] = data['words_uniq'] / 3000  # кол-во уникальных слов на 3 тыс. слов
        #data['words_uniq_10k'] = data['words_uniq'] / 10000 #кол-во уникальных слов на 10 тыс. слов

        y = None
        if mode == 'train':
            y = data['idauthor']
        X = data.drop(columns = columns2drop)

        #Add PCA features
        n_components = 4
        pca_cols2drop = ['sentence_mean', 'uniq_per_sent_chunk', 'uniq_per_words_chunk']
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
        model = xgb.XGBClassifier(n_estimators = 400, max_depth = 24, colsample = 1, subsample = 1, seed = seed)
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
                   'sentences_text', 'words_text','sentence_mean', \
                   'sentences_chunk', 'words_chunk',
                   'words_uniq_chunk','uniq_per_sent_chunk','uniq_per_words_chunk', \
                  'NOUN','ADJF','ADJS','COMP','VERB','INFN','PRTF','PRTS','GRND','NUMR',\
                  'ADVB','NPRO','PRED','PREP','CONJ','PRCL','INTJ', 'predict']
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
            int_cols = ['idtext', 'idchunk', 'idauthor', 'sentences_text', 'words_text', 'sentences_chunk', 'words_chunk', 'words_uniq_chunk']
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
        df_stat.loc[aidtext, 'predict'] = y_res.astype(int)
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
        group = group.agg({'sentences_text': ['mean'],
                           'words_text': ['mean'],
                           'sentence_mean': ['mean'],
                           'sentences_chunk': ['mean'],
                           'words_chunk': ['mean'],
                           'words_uniq_chunk': ['mean'],
                           'uniq_per_sent_chunk': ['mean'],
                           'uniq_per_words_chunk': ['mean'],
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
        if mode == 'test':
            data['predict'] = data['predict'].astype(int)
        data = pd.merge(data, authors, left_on='predict', right_on='idauthor', how='left', suffixes=('', '_predict'))
        return data

    def vizualize2d(self, mode='train'):
        n_components = 2
        env = Environment()
        data = self.get_texts_stat(mode = mode)
        columns = data.columns
        #print(data)
        #print(columns)
        columns2drop = ['idtext', 'idauthor', 'author', 'name',
                        'sentences_text', 'words_text', 'sentence_mean', 'sentences_chunk', 'words_chunk',
                        'words_uniq_chunk', 'uniq_per_sent_chunk', 'predict', 'shortname', 'name_author']

        y = data['idauthor'].values
        X = data.drop(columns = columns2drop).values
        #print(y, X)
        #return 0

        #print(data)
        #print(X, y)
        pca = PCA(n_components = n_components)
        #pca = TSNE(n_components=2)
        X_new = pca.fit_transform(X, y)
        print('PCA ratio 2 components', pca.explained_variance_ratio_)
        #print('components', pca.components_)
        #print(X_new)
        tdf = pd.DataFrame(data = X_new, columns = ['PC1', 'PC2'])
        finalDf = pd.concat([tdf, data[['idauthor','name','shortname']]], axis = 1)
        print('dataframe ', finalDf)

        mpl.style.use('default')

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Tahoma']

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Component 1. Вклад '+str(round(pca.explained_variance_ratio_[0],2)), fontsize=12)
        ax.set_ylabel('Component 2. Вклад '+str(round(pca.explained_variance_ratio_[1],2)), fontsize=12)
        ax.set_title('2 component PCA. Точность '+str(round(sum(float(i) for i in pca.explained_variance_ratio_),2)), fontsize=12)
        targets = data.idauthor.unique()
        print(targets)
        legends = data.shortname.unique()
        print(legends)
        #print(targets)
        #colors = ['r', 'g', 'b']
        #colors = "bgcmykw" #without r
        #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        #              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        #              '#bcbd22', '#17becf']
        colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
                  "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
                  "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
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