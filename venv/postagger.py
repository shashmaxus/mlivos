import pandas as pd
import numpy as np
import pickle

from common import Environment
from sharedsvc import Word_Encoder
from corpus import OpenCorpus

from timeit import default_timer as timer

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
from matplotlib import rcParams

from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.io as pio

import plotly.graph_objs as go
import plotly.plotly as py
from plotly.graph_objs import *

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, ColorBar, CategoricalColorMapper
from bokeh.transform import linear_cmap
from bokeh.transform import factor_cmap
from bokeh.palettes import Viridis3, Viridis256, Spectral6, d3, all_palettes, mpl
from bokeh.io import export_png

class POSTagger:

    #Return part-of-speech by dictionary
    def pos_word_by_voc(self, awords):
        corpus = OpenCorpus()
        voc = corpus.vocabulary()
        r_words = []
        for word in awords:
            try:
                r_gram=voc[voc['word'] == word]['gram'].values[0]
            except:
                r_gram=''
            r_words.append(r_gram)
        return (r_words)

    # Return part-of-speech by Vocabulary search (massive)
    def pos_by_voc(self, awords):
        corpus = OpenCorpus()
        voc = corpus.vocabulary()
        df_voc = voc[voc['word'].isin(awords)]
        r_words = []
        for word in awords:
            try:
                r_gram = df_voc[df_voc['word'] == word]['gram'].values[0]
            except:
                r_gram=''
            r_words.append(r_gram)
        return (r_words)

    # Return part-of-speech by machine learning
    def pos_word_by_ml(self, awords):
        env = Environment()
        enc = Word_Encoder()
        file_model = env.filename_model_tree()
        clf = pickle.load(open(file_model, 'rb'))
        a_predict = np.array([enc.word2token('')])
        for word in awords:
            a_padd = [enc.word2token(word)]
            #print(word, a_padd)
            a_predict = np.append(a_predict, a_padd, axis=0)
        a_predict = a_predict[1:]
        #print(a_predict[0, 100])
        predictions = clf.predict(a_predict[:, 0:])
        return(predictions[0:])

    #Fill dataframe with part-of-speech
    def pos(self, df, mode_fast=True, use_cache=True):
        env = Environment()
        enc = Word_Encoder()
        df_res = df
        t_start = timer()

        c = OpenCorpus()
        g = c.grammemes()
        dg = g.to_dict().get('name')

        a_predict = np.array([enc.word2token('')])
        #a_words = ['']
        n_words = df_res.shape[0]

        env.debug(1, ['POStagger', 'pos', 'START Vocabulary prediction %s words' % n_words])
        a_words = df_res['word'].tolist()
        a_ml_words = []
        predictions_voc = self.pos_by_voc(a_words)
        p_se = pd.Series(predictions_voc)
        df_res['gram'] = p_se.values
        df_res['gram_voc'] = p_se.values
        df_res['gram_ml'] = ''
        t_end = timer()
        env.debug(1, ['POStagger', 'pos', 'END Vocabulary prediction %s sec.' % env.job_time(t_start, t_end)])
        #print(predictions_voc)

        if mode_fast:
            #env.debug(1, ['POStagger', 'pos', 'START Fast mode vocabulary search. Words %s' % df.shape[0]])
            df_ni_voc = df_res[df_res['gram_voc']=='']
            n_words = df_ni_voc.shape[0]
        else:
            df_ni_voc = df_res
        #print('non-vocabulary',df_ni_voc)
        if not df_ni_voc.empty:
            env.debug(1, ['POStagger', 'pos', 'START Encoding %s words' % n_words])
            for index, serie in df_ni_voc.iterrows():
                word = df_ni_voc.at[index, 'word']
                #print(word)
                a_padd = np.array([enc.word2token(word)])
                a_predict = np.append(a_predict, a_padd, axis=0)
                a_ml_words.append(word)
                #print(a_words, a_predict)
            a_predict = a_predict[1:, :]
            #print(a_predict)
            #print('ml_words',a_ml_words)
            t_end = timer()
            env.debug(1, ['POStagger', 'pos', 'END Encoding %s words %s sec.' % (n_words, env.job_time(t_start, t_end))])

        t_start = timer()
        env.debug(1, ['POStagger', 'pos', 'START Model prediction'])
        clf = pickle.load(open(env.filename_model_tree(), 'rb'))
        predictions_ml = clf.predict(a_predict[:, 0:])
        # print('ml', predictions_ml)
        t_end = timer()
        env.debug(1, ['POStagger', 'pos', 'END Model prediction %s sec.' % env.job_time(t_start, t_end)])
        #print('ml_words_prediction',list(zip(a_ml_words,predictions_ml)))

        t_start = timer()
        i = 0
        s_pvoc=''
        s_pml=''
        for index, row in df_res.iterrows():
            word = df_res.at[index, 'word']
            s_pvoc = df_res.at[index, 'gram_voc']
            #s_pvoc = predictions_voc[i]
            #print('s_pvoc', word, s_pvoc)
            #df_res.at[index, 'gram_voc'] = s_pvoc
            if s_pvoc == '':
                if mode_fast:
                    try:
                        j = a_ml_words.index(word)
                    except:
                        pass
                    else:
                        s_pml = dg.get(predictions_ml[j])
                        #print(word,s_pml)
                else:
                    s_pml = dg.get(predictions_ml[i])
                df_res.at[index, 'gram_ml'] = s_pml
                df_res.at[index, 'gram'] = s_pml
            i = i + 1
        t_end = timer()
        env.debug(1, ['POStagger', 'pos', 'ML predictions dataframe filled %s sec' % env.job_time(t_start, t_end)])
        #print(df_res)
        return df_res

    #Transform Words Dataframe to Tokenz DataFrame
    def tokenize(self, dftokenz = pd.DataFrame(), persistent=True, n_frac=1):
        env = Environment()
        enc = Word_Encoder()
        t_start = timer()
        if dftokenz.empty:
            dftokenz = self.tokenz()
        if n_frac<1:
            dftokenz = dftokenz.sample(frac=n_frac)
        env.debug(1, ['Transforming to tokenz: START %s words' % dftokenz.shape[0]])

        gmask = dftokenz.groupby(['gram'])
        df_posstat = gmask.count()
        df_posstat.to_csv(env.filename_stat_pos_tokenz_csv(), encoding='utf-8')
        print('POSTagger','train dataset stat:\n',gmask.count())

        fields = ['s_suffix2', 's_suffix3', 's_prefix2', 's_prefix3', 'n_token', 'n_len', 'n_tokens2', 'n_tokens3',
                  'n_tokenp2', 'n_tokenp3']

        for field in fields:
            val = 0.0
            if field[0] == 's':
                val = ''
            dftokenz[field] = val

        n_letters = 0
        s_letters = env.list_rus_letters()
        di_letters = env.di_bgm_byletters
        #bgm_columns_i = env.bgm_columns_list(mode=0)
        bgm_columns = env.bgm_columns_list(mode=1)

        #print('bgm_columns', bgm_columns)
        for column_name in bgm_columns:
            dftokenz[column_name] = None

        t_end = timer()
        env.debug(1, ['POStagger','Letters bigram columns added', env.job_time(t_start, t_end)])

        #Form tokenz
        t_start = timer()
        for index, serie in dftokenz.iterrows():
            # print (serie.values)
            a_word = enc.s2token(index, serie)
            i = 2
            # print(a_word)
            for field in fields:
                dftokenz.at[index, field] = a_word[i]
                # print(field, a_word[i])
                i = i + 1
            # print(dftokenz.loc[index])
            #Letters bigram binaries
            for n_l in range(0, len(a_word[0])-1):
                n_l2 = n_l +1
                di_n = di_letters.get('%s%s' % (a_word[0][n_l], a_word[0][n_l2]))
                if di_n is not None:
                    #print(di_n)
                    #print(bgm_columns[di_n])
                    dftokenz.at[index, bgm_columns[di_n]] = 1
        t_end = timer()
        env.debug(1, ['Transforming to tokenz: COMPLETE',env.job_time(t_start, t_end)])
        if persistent:
            dftokenz.to_csv(env.filename_tokenz_csv(), encoding='utf-8')
            env.debug(1, ['Tokenz written to CSV:', env.filename_tokenz_csv()])
        return dftokenz

    # Form statistic
    def tokenz_create_stat(self, dftokenz=pd.DataFrame(), n_frac=1):
        env = Environment()
        enc = Word_Encoder()
        di_letters=Environment.di_bgm_byletters
        bgm_columns = env.bgm_columns_list(mode=1)
        t_start = timer()
        if dftokenz.empty:
            dftokenz = self.tokenz()
        if n_frac < 1:
            dftokenz = dftokenz.sample(frac=n_frac)
        env.debug(1, ['POStagger','create_stat', 'Collecting statistic START %s words' % dftokenz.shape[0]])
        di_tokenz_stat = (dftokenz.count()).to_dict()
        di_tokenz_res = {}
        #print('di_letters', di_letters)
        print('di_tokenz_stat', di_tokenz_stat)
        bgm_astat = [['init',0]]
        bgm_index = []
        for key in di_letters:
            di_n = di_letters.get(key)
            column_stat = di_tokenz_stat.get(bgm_columns[di_n])
            #di_tokenz_res[key] = column_stat
            bgm_astat.append([key, column_stat])
            bgm_index.append(di_n)
        bgm_astat = bgm_astat[1:]
        print('column stat', bgm_astat)
        df_bgm_stat = pd.DataFrame(data=bgm_astat, columns=['bigram', 'counts'], index=bgm_index)
        df_bgm_stat.index.name = 'idbigram'
        df_bgm_stat = df_bgm_stat.sort_values(by=['counts'], ascending=False)
        print('bgm_stat\n', df_bgm_stat)
        df_bgm_stat.to_csv(env.filename_stat_bigram_letters_csv(), encoding='utf-8')

    # Return tokenz dataset
    def tokenz(self):
        env = Environment()
        df_tokenz = pd.DataFrame()
        file_tokenz = env.filename_tokenz_csv()
        try:
            df_tokenz = pd.read_csv(file_tokenz, index_col='idcorpus', encoding='utf-8')
        except:
            env.debug(1, ['Failed to read tokenz file:', file_tokenz])
        else:
            env.debug(1, ['Read tokenz OK:', file_tokenz])
        return df_tokenz

    #Train model
    def train(self, df = pd.DataFrame(), b_cv = True, n_splits = 5, b_smoketest = True, n_frac=1):
        env = Environment()
        enc = Word_Encoder()
        df_train=df
        bgm_columns = env.bgm_columns_list(mode=1)
        drop_columns = ['word', 'gram', 's_suffix2', 's_suffix3',
                        's_prefix2', 's_prefix3', 'n_token'] #, 'bgm_l_None'
        #drop_columns.extend(['bgm_l_%s' % (i) for i in range(1, env.bgm_columns_max()) if 'bgm_l_%s' % (i) not in bgm_columns])
        env.debug(1, ['POStagger','train','Drop colums: %s' % (drop_columns)])

        if df_train.empty:
            t_start = timer()
            df_train=self.tokenz()
            t_end = timer()
            env.debug(1, ['POSTagger','train','tokenz loaded:', 'time:', env.job_time(t_start, t_end)])

        t_start = timer()
        env.debug(1, ['POStagger','train','Learning: START'])
        if n_frac<1:
            df_train = df_train.sample(frac=n_frac)
            #print(df_train.shape)

        #df_train2 = df_train[bgm_columns]
        #print(df_train2.shape)
        #df_train2 = df_train2.astype({"idgram": int})
        df_train=df_train.drop(columns=drop_columns, axis=1)
        env.debug(1, ['POStagger','Train colums: %s' % (df_train.columns.tolist())])

        #df_train = df_train.drop_duplicates() #slow-slow
        #print(df_train.head())

        df_train = df_train.fillna(0)
        file_x = env.filename_xtrain_csv()
        df_train.to_csv(file_x, encoding='utf-8')
        env.debug(1, ['POStagger','train','Save X',file_x])
        array = df_train.values
        #print(df_train)
        X = array[:, 1:]
        #Y = array[:, 0]
        Y = df_train['idgram'].values
        #print(X, Y)
        #validation_size = 0.20
        seed = 241

        sc = StandardScaler()
        #Y_sc = sc.fit_transform(Y)

        if b_cv: #Need cross-validation
            scoring = 'accuracy'
            # scoring = 'f1_samples'
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            t2_start = timer()
            if True: #Decision tree
                env.debug(1, ['Tree cross-validation'])
                # clf = DecisionTreeClassifier(criterion='gini', random_state=seed)  # 0.79
                # clf = KNeighborsClassifier(n_neighbors=230)
                clf = DecisionTreeClassifier(criterion='entropy', random_state=seed)  # 0.81
                env.debug(1, ['Calculate cross_val_score. Splits=%s' % (n_splits)])
                scores = cross_val_score(clf, X, Y, cv=kf)
                print('DTree scores:',scores.mean(),'raw',scores)

            if False: #Logistic regression
                env.debug(1, ['LGR cross-validation'])
                n_Cs = [0.01]
                X = array[:, 5:]
                X_sc = sc.fit_transform(X)
                Y = df_train['idgram'].values
                Y [Y>0] = 1
                print(X_sc, Y)
                for n_c in n_Cs:
                    #clf = LogisticRegression(penalty='l2', solver='saga', C=n_c, multi_class='multinomial')
                    clf = LogisticRegression(penalty='l2', solver='liblinear', C=n_c)
                    # clf = SVC(kernel='linear', C=10000, random_state=241)
                    # clf = SVC(kernel='linear', C=0.01, random_state=seed)
                    # clf = SVC(random_state=seed)
                    # clf = Perceptron()
                    env.debug(1, ['Calculate cross_val_score. Splits=%s C=%s' % (n_splits, n_c)])
                    scores = cross_val_score(clf, X_sc, Y, cv=kf)
                    print(scores)

            if False: #GBM, RandomForest
                env.debug(1, ['GBM cross-validation'])
                asteps=[20] #GBM
                #asteps=[100] #RandomForest
                for i in asteps:
                    #clf = RandomForestClassifier(n_estimators=i)
                    clf = GradientBoostingClassifier(n_estimators=i, max_depth=8) #, max_features='sqrt'
                    env.debug(1, ['Calculate cross_val_score. Splits=%s Estimators=%s' % (n_splits, i)])
                    scores = cross_val_score(clf, X, Y, cv=kf)
                    print(scores)

            if False: #XGBoost
                env.debug(1, ['XGboost cross-validation'])
                asteps = [2]  # n_estimators
                # asteps=[100] #RandomForest
                for i in asteps:
                    clf = xgb.XGBClassifier(n_estimators=i, max_depth=6)  # , max_features='sqrt'
                    env.debug(1, ['POStagger','train','Calculate cross_val_score. Splits=%s Estimators=%s' % (n_splits, i)])
                    scores = cross_val_score(clf, X, Y, cv=kf)
                    print(scores)

            #X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
            #
            #
            #                                                                           random_state=seed)
        t2_end = timer()
        t_end = timer()
        env.debug(1, ['CV completed:', 'time:', env.job_time(t_start, t_end)])

        #Training
        X_train, Y_train = X, Y
        # model = SVC()
        # model= DecisionTreeClassifier() #79
        # model= LinearDiscriminantAnalysis() #47
        # model=LogisticRegression() #48
        # model = KNeighborsClassifier(n_neighbors=200) #48
        # model = GaussianNB()   #43
        #print('Fit...')

        #print('Validate...')
        # predictions = model.predict(X_validation)

        # print(accuracy_score(Y_validation, predictions))
        # print(confusion_matrix(Y_validation, predictions))
        # print(classification_report(Y_validation, predictions))

        t_start = timer()
        env.debug(1, ['Training: START'])
        clf.fit(X_train, Y_train)
        t_end = timer()
        env.debug(1, ['Training: END',env.job_time(t_start, t_end)])

        pickle.dump(sc, open(env.filename_scaler(), 'wb'))
        pickle.dump(clf, open(env.filename_model_tree(), 'wb'))

        # Smoke test
        if b_smoketest:
            X_smoke_predict = ['съеште', 'ещё', 'этих', 'мягких',
                         'французских', 'булок']
            a_smoke = np.array([enc.word2token(elem) for elem in X_smoke_predict])
            y_predictions = clf.predict(a_smoke[:, 0:])
            y_predictions_proba = clf.predict(a_smoke[:, 0:])
            #print(y_predictions)
            print('Prediction',list(zip(X_smoke_predict, y_predictions)))
            print('Proba', list(zip(X_smoke_predict, y_predictions_proba)))
        return clf

    #test on corpus file
    def test(self, n_min=1, n_max=1):
        t_start = timer()
        env = Environment()
        df_test = pd.DataFrame()
        for i in range(n_min, n_max + 1):
            try:
                dffile = pd.read_csv(env.filename_corpus_csv(i), index_col='idcorpus', encoding='utf-8')
            except:
                env.debug(1, ['POStagger', 'test', 'Failed to read corpus file:', env.filename_corpus_csv(i)])
            else:
                env.debug(1, ['POStagger', 'test', 'Read OK:', env.filename_corpus_csv(i)])
                if not dffile.empty:
                    df_test = df_test.append(dffile)
        df_test = df_test.drop_duplicates()
        df_test.columns = ['word', 'gram', 'idgram']
        df_test = df_test.reset_index(drop=True)
        df_test.index.name = 'idcorpus'
        df_test['gram_valid'] = df_test['gram']
        n_testsize = df_test.shape[0]
        env.debug(1, ['POStagger','test','START %s words' % n_testsize])
        df_test=self.pos(df_test)
        print('Test result', df_test)
        df_err=df_test[df_test['gram_valid'] != df_test['gram']]
        print('Test errors:',df_err)
        df_err.to_csv(env.filename_test_err_csv(), encoding='utf-8')
        env.debug(1, ['POStagger', 'test', 'test accuracy %s' % (1 - df_err.shape[0] / n_testsize)])
        t_end = timer()
        env.debug(1, ['POSTagger', 'test', 'test time:', env.job_time(t_start, t_end),'sec.'])


    def vizualize2d(self, n_frac=0.01, b_annotations=False):
        n_components = 2
        env = Environment()
        c = OpenCorpus()
        di_g = c.grammemes(mode=1)
        data = self.tokenz().sample(frac=n_frac)

        data = data.fillna(0)
        #print(data['idgram'].shape)
        #print(data.index.shape)
        tdf = pd.DataFrame(index=data.index)
        tdf['idgram'] = data['idgram']
        tdf['gram'] = data['gram']
        tdf['word'] = data['word']
        #print(tdf)

        drop_columns = ['word', 'gram', 's_suffix2', 's_suffix3',
                        's_prefix2', 's_prefix3', 'n_token']  # , 'bgm_l_None'
        # drop_columns.extend(['bgm_l_%s' % (i) for i in range(1, env.bgm_columns_max()) if 'bgm_l_%s' % (i) not in bgm_columns])
        env.debug(1, ['POStagger', 'visualize2D', 'Drop colums: %s' % (drop_columns)])
        data = data.drop(columns=drop_columns, axis=1)
        values = data.values
        X = values[:, 1:]
        y = values[:, 0]
        #print(data.head,X, y)
        #return 0

        #Scalers
        sc = StandardScaler()
        min_max_scaler = preprocessing.MinMaxScaler()
        max_abs_scaler = preprocessing.MaxAbsScaler()
        #X = sc.fit_transform(X)

        #PCA
        b_pca = False
        b_sne = True
        if b_pca:
            model = PCA(n_components=n_components)
        if b_sne:
            model = MDS(n_components=n_components) #TSNE
        X_new = model.fit_transform(X, y)
        if b_pca:
            print('PCA ratio',n_components,'components', model.explained_variance_ratio_)
        #X_new = sc.fit_transform(X_new)
        #X_new = preprocessing.scale(X_new)
        if b_pca:
            X_new=max_abs_scaler.fit_transform(X_new)
        #return 0

        #tdf = pd.DataFrame(data=X_new, columns=['PC1', 'PC2'], index=data.index)
        tdf['PC1'] = X_new[:, 0]
        tdf['PC2'] = X_new[:, 1]
        #finalDf = pd.concat([tdf, data[['idgram']]], axis=1)
        df_groups = tdf.groupby('idgram').count()
        #print(df_groups)
        #return 0
        tdf['counts'] = 0
        for index, serie in tdf.iterrows():
            n_idgram = tdf.at[index, 'idgram']
            tdf.at[index, 'counts'] = df_groups[df_groups.index == n_idgram]['gram']
        tdf = tdf.sort_values(by=['counts'], ascending=False)
        #print(tdf)

        #Draw
        i = 0
        N = df_groups.shape[0]
        s_title = ''
        if b_pca:
            s_title = '2 component PCA. Точность %s' % (round(sum(float(i) for i in model.explained_variance_ratio_), 2))
        if b_sne:
            s_title= 't-SNE'

        #Plotly
        if False: #Plotly
            py.sign_in('shashmaxus', 'AdfwTulrOoV3cSlbZT3B')
            c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N)]
            data_trace = []
            for index, row in df_groups.iterrows():
                #print(index)
                df_trace = tdf[tdf['idgram'] == index]
                #print(df_trace)
                g_trace = go.Scatter (x = df_trace['PC1'].values,
                                      y = df_trace['PC2'].values,
                                      name = df_trace['gram'].values[0],
                                      mode = 'markers',#'markers+text'
                                      marker = dict(size=8,
                                                  color=i,#c[i]
                                                  opacity=0.8,
                                                    colorscale='Viridis'),
                                      text=df_trace['word'],
                                      textfont = dict(family='sans serif',
                                                      size=12)
                                        )
                data_trace.append(g_trace)
                i+=1
            layout = go.Layout(
                title=s_title_pca,
                xaxis=dict(title=('Component 1. Вклад %s' % (round(pca.explained_variance_ratio_[0],2)))),
                yaxis=dict(title=('Component 2. Вклад %s' % (round(pca.explained_variance_ratio_[1],2)))))
            fig2 = go.Figure(data=data_trace, layout=layout)
            py.image.save_as(fig2, filename='c:/prj/mlivos_data/temp/Words2.png')

        #Bokeh
        if True:
            palette = d3['Category20'][len(tdf['gram'].unique())]
            #palette = all_palettes['Category20'][len(tdf['gram'].unique())]
            #palette = Viridis256[len(tdf['gram'].unique())]
            #palette = Viridis256
            color_map = CategoricalColorMapper(factors=tdf['gram'].unique(),
                                                   palette = palette)
            #print(mapper)
            fig = figure(title=s_title, toolbar_location=None)
            source = ColumnDataSource(tdf[['gram', 'PC1', 'PC2']])
            fig.scatter(x='PC1',
                        y='PC2',
                        size = 12,
                        color={'field': 'gram', 'transform': color_map},
                        legend='gram', source=source)
            show(fig)
            export_png(fig, filename="c:/prj/mlivos_data/temp/PCA.png")
        return 0