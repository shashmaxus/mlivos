import pandas as pd
import nltk

from common import Environment
from corpus import OpenCorpus
from postagger import POSTagger
from analyzer import mlAnalyzer
from sharedsvc import Word_Encoder
from reports import Reporter

def main():
    pd.set_option("display.max_columns", 100)
    pd.set_option('display.width', 1000)

    env = Environment()
    c = OpenCorpus()
    t = POSTagger()
    a = mlAnalyzer()
    enc = Word_Encoder()
    r = Reporter()
    #c.dict_xml2csv(lines=100000)
    #c.grammemes_xml2csv()
    #c.vocabulary_from_corpus(1,1000)
    g = pd.DataFrame()
    g = c.grammemes()
    #dg = g.to_dict().get('name')
    dg = c.grammemes(mode=1) #grammemes by id
    da = c.authors(mode=1) # authors by id

    #print(dg)
    #print(p.head())
    #for i in range(2015,3000):
    #    c.corpus_xml2csv(i)
    #c.corpus_xml2csv(2)
    #for i in range (125,150):
    #    c.corpus_xml2txt(i)
    #print(c.vocabulary_from_corpus(1,2000).head())
    #voc=c.vocabulary()
    #print(voc.head())
    #t.tokenize()
    #print(t.tokenize(voc, n_frac=1))
    #t.tokenz_create_stat()
    #print(env.bgm_stat())
    #print(t.tokenz())
    #print(c.vocabulary())
    #print(enc.word2token('паровоз'))
    #print(enc.word2token('аз'))
    #t.train(n_frac=0.8)
    #t.test(2000,2048)
    #a.process_from_texts_file([35,35])
    #arrt = [2, 45, 43, 44, 42, 40, 41, 46, 36, 37, 38, 34]
    #arrt = [2]
    #for i in range (31,48):
    #for i in arrt:
    #    a.process_from_texts_file([i])
    #t.vizualize2d(n_frac=0.01)
    #nltk.download()
    #a.vizualize2d()
    # a.vizualize2d(mode='test')
    #a.model_train()
    #return 0
    #y = a.predict([0, 1, 2, 3, 4])
    #y = a.predict([0, 1, 2, 3, 4])
    #print(a.predict([0,1, 2,3,4], b_makestat=True))
    #for i in y:
    #    print('idtext=%s' % i, da.get(i))
    text2predict = [4]
    #y = a.predict(text2predict)  # предсказать - указать номер текста
    #j = 0
    #for i in y:
    #    print('idtext=%s' % text2predict[j], 'Автор=%s (%s)' % (i, da.get(i)))
    #    j = j + 1

    #predict=(t.pos_word_by_voc(['съеште', 'школа','господина','приехал',
    #                       'глокая','куздра','штеко','будланула','бокра','и','кудрячит','бокрёнка']))

    #X_predict=['съеште', 'школа', 'господина', 'приехал',
    # 'глокая', 'куздра', 'штеко', 'будланула', 'бокра', 'и', 'кудрячит', 'бокрёнка',
    #    'он', 'видел', 'их', 'семью', 'своими',  'глазами']
    #X_predict=['символ']
    #y_predict=t.pos_word_by_ml(X_predict)

    #print(['%s/%s' % (X_predict[i],dg.get(y_predict[i])) for i in range(0,len(y_predict))])
    r.make_report()

main()