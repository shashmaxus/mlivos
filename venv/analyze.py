import pandas as pd
import numpy as np
import pickle
from sharedsvc import Word_Encoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

def main():
    enc = Word_Encoder()
    file_model = 'c:/prj/temp/clf.model'
    #file_txt = 'c:/prj/corpus/txt/2.txt'
    #file_txt = 'c:/prj/corpus/txt/4.txt'
    #file_txt = 'c:/prj/texts/pushkin_dubrovsky.txt'
    #file_txt = 'c:/prj/texts/pushkin_kap_dochka.txt'
    #file_txt = 'c:/prj/texts/lermontov_geroy.txt'
    #file_txt = 'c:/prj/texts/lermontov_mtsiri.txt'

    file_gram = 'c:/prj/temp/dfgram.csv'
    file_res = 'c:/prj/temp/dfres.csv'

    dfgram = pd.read_csv(file_gram, index_col='idgram', encoding='utf-8')
    dfres = pd.read_csv(file_res, index_col='idtext', encoding='utf-8')

    file=open(file_txt,'r')
    text=[file.read()]
    file.close()
    #print(text)
    vectorizer = CountVectorizer()
    #analyze=vectorizer.build_analyzer()
    #print(analyze(text))

    X = vectorizer.fit_transform(text)
    #print(vectorizer.get_feature_names())
    #print(X)
    #vector = vectorizer.transform(text)
    #print(vector)

    # чтобы узнать количественное вхождение каждого слова:
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    #final_matrix = np.array([np.array(vectorizer.get_feature_names()), matrix_freq])
    #final_matrix=list(zip(vectorizer.get_feature_names(),matrix_freq))
    #print(final_matrix)
    #words = vectorizer.vocabulary_
    #for word in words.keys():
    #    print(word,words.get(word))
    #vector = vectorizer.transform(text)
    data_cols = ['word', 'count']
    data=pd.DataFrame(columns=data_cols)
    zl=zip(vectorizer.get_feature_names(),matrix_freq)
    a_predict=np.array([enc.word2token('')])
    #print(a_predict)
    for feature, count in zl :
        s = pd.Series(data=[feature, count],index=data_cols)
        data = data.append(s, ignore_index=True)
        a_padd = np.array([enc.word2token(feature)])
        #print(a_padd)
        #print(a_predict.shape)
        #print(a_padd.shape)
        a_predict=np.append(a_predict, a_padd, axis=0)
        #a_predict=np.concatenate((a_predict,a_padd))
    data['gram']=''
    #print(data.head(1000))
    #print(a_predict)

    clf = pickle.load(open(file_model, 'rb'))
    predictions = clf.predict(a_predict[:, 6:12])
    #print(predictions)
    i=1
    for index, row in data.iterrows():
        data.at[index,'gram'] = predictions[i]
        i=i+1
    #print(data.head(1000))
    n_size = data['count'].sum()
    #print(n_size)
    grouped = data.groupby(['gram'])
    #print(grouped.groups)
    #serie_stat=grouped['gram'].count()/n_size
    serie_stat = grouped['count'].sum()/n_size
    #print(serie_stat)
    #print(serie_stat.sum())
    #return 0
    name_text=file_txt
    s_res=pd.Series([])
    s_res['name'] = name_text
    for index, row in dfgram.iterrows():
        name_col=dfgram.at[index, 'name']
        try:
            s_res[name_col] = serie_stat[name_col]
        except:
            s_res[name_col] = 0
        #print(name_col, serie_stat[name_col])
    #print(s_res)
    dfres = dfres.append(s_res, ignore_index=True)
    #print(dfres.head())
    dfres = dfres.reset_index(drop=True)
    dfres.index.name = 'idtext'
    dfres.to_csv(file_res, encoding='utf-8')

    #print(grouped['gram'].agg({'count' : count / n_size}))
            #print(row)
    #print(list(zip(a_predict[:, 0], predictions)))

    #predictions = clf.predict(a_smoke[:, 6:12])
    #a_smoke = np.array(
    #    [enc.word2token('съеште'), enc.word2token('ещё'), enc.word2token('этих'),
    #     enc.word2token('мягких'), enc.word2token('французских'),enc.word2token('булок')])
    # print(a_smoke)
    #predictions = clf.predict(a_smoke[:, 6:12])
    #print(list(zip(a_smoke[:, 0], predictions)))

main()