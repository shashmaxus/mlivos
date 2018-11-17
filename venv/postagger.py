import pandas as pd
import numpy as np
import pickle
from common import Environment
from sharedsvc import Word_Encoder
from corpus import OpenCorpus

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

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
                r_gram=df_voc[df_voc['word'] == word]['gram'].values[0]
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
            a_predict = np.append(a_predict, a_padd, axis=0)
        #print(a_predict)
        predictions = clf.predict(a_predict[:, 7:12])
        return(predictions[1:])

    #Fill dataframe with part-of-speech
    def pos(self, df):
        env = Environment()
        enc = Word_Encoder()
        df_res = df
        a_predict = np.array([enc.word2token('')])
        for index, serie in df_res.iterrows():
            word = df_res.at[index, 'word']
            a_padd = np.array([enc.word2token(word)])
            a_predict = np.append(a_predict, a_padd, axis=0)

        predictions_voc = self.pos_by_voc(a_predict[:, 0])
        # print('voc', predictions_voc)
        clf = pickle.load(open(env.filename_model_tree(), 'rb'))
        predictions_ml = clf.predict(a_predict[:, 7:12])
        # print('ml', predictions_ml)
        i = 1
        for index, row in df_res.iterrows():
            df_res.at[index, 'gram_voc'] = predictions_voc[i]
            df_res.at[index, 'gram_ml'] = predictions_ml[i]
            if predictions_voc[i] != '':
                df_res.at[index, 'gram'] = predictions_voc[i]
            else:
                df_res.at[index, 'gram'] = predictions_ml[i]
            i = i + 1
        return df_res

    #Transform Words Dataframe to Tokenz DataFrame
    def tokenize(self, df, persistent=True):
        env = Environment()
        env.debug(1, ['Transforming to tokenz: START'])
        fields = ['s_suffix2', 's_suffix3', 's_prefix2', 's_prefix3', 'n_token', 'n_len', 'n_tokens2', 'n_tokens3',
                  'n_tokenp2', 'n_tokenp3']
        dftokenz = df
        enc = Word_Encoder()
        for field in fields:
            val = 0.0
            if field[0] == 's':
                val = ''
            dftokenz[field] = val

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
        # print(dftokenz.head(100))
        env.debug(1, ['Transforming to tokenz: COMPLETE'])
        if persistent:
            file_tokenz=env.filename_tokenz_csv()
            env.debug(1, ['Write tokenz to CSV:', file_tokenz])
            dftokenz.to_csv(file_tokenz, encoding='utf-8')
        return dftokenz

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
    def train(self, df=pd.DataFrame()):
        env = Environment()
        n_splits=5
        df_train=df
        if df_train.empty:
            df_train=self.tokenz()
        env.debug(1, ['Learning: START'])
        #print(df_train.head(100))
        df_train2 = df_train[['gram', 'n_len', 'n_tokens2', 'n_tokens3',
                  'n_tokenp2', 'n_tokenp3']]
        #print(df_train2.shape)
        df_train2 = df_train2.drop_duplicates()
        #print(df_train2.shape)
        file_x = env.filename_xtrain_csv()
        df_train2.to_csv(file_x, encoding='utf-8')
        env.debug(1, ['Save X',file_x])
        array = df_train2.values
        X = array[:, 1:]
        Y = array[:, 0]
        #print(X, Y)
        #validation_size = 0.20
        seed = 241
        scoring = 'accuracy'
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # clf = DecisionTreeClassifier(criterion='entropy', random_state=seed) #0.81
        clf = DecisionTreeClassifier(criterion='gini', random_state=seed)  # 0.79
        # clf = SVC(kernel='linear', C=10000, random_state=241)
        # clf = SVC(kernel='linear', C=0.01, random_state=seed)
        # clf = SVC(random_state=seed)
        # clf = LogisticRegression()
        # clf = Perceptron()
        env.debug(1, ['Calculate cross_val_score. Splits=',str(n_splits)])
        scores = cross_val_score(clf, X, Y, cv=kf)
        print(scores)
        #X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
        #                                                                                random_state=seed)
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

        env.debug(1, ['Training: START'])
        clf.fit(X_train, Y_train)
        env.debug(1, ['Training: END'])
        filename = env.filename_model_tree()
        pickle.dump(clf, open(filename, 'wb'))

        # Smoke test
        b_smoketest = True
        if b_smoketest:
            enc = Word_Encoder()
            a_smoke = np.array([enc.word2token('съеште'), enc.word2token('ещё'), enc.word2token('этих'),
                            enc.word2token('мягких'), enc.word2token('французских'), enc.word2token('булок')])
            # print(a_smoke)
            predictions = clf.predict(a_smoke[:, 7:12])
            print(list(zip(a_smoke[:, 0], predictions)))
        return clf
