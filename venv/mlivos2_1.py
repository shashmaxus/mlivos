import pandas as pd
from common import Environment
from corpus import OpenCorpus
from postagger import POSTagger
from analyzer import mlAnalyzer
from sharedsvc import Word_Encoder

def main():
    env = Environment()
    c = OpenCorpus()
    t = POSTagger()
    a = mlAnalyzer()
    enc = Word_Encoder()
    #c.dict_xml2csv(lines=10000)

    #c.grammemes_xml2csv()
    #c.vocabulary_from_corpus(1,1000)
    g = pd.DataFrame()
    g = c.grammemes()
    dg = g.to_dict().get('name')
    #print(p.head())
    #for i in range(2015,3000):
    #    c.corpus_xml2csv(i)
    #c.corpus_xml2csv(2)
    #for i in range (125,150):
    #    c.corpus_xml2txt(i)
    #print(c.vocabulary_from_corpus(1,2000).head())
    voc=c.vocabulary()
    #print(voc.head())
    #t.tokenize()
    #print(t.tokenize(voc, n_frac=1))
    #t.tokenz_create_stat()
    #print(env.bgm_stat())
    #print(t.tokenz())
    #print(c.vocabulary())
    #print(enc.word2token('паровоз'))
    #print(enc.word2token('аз'))
    t.train(n_frac=0.8)
    #t.test(2048,2048)
    #a.process_from_texts_file([32,33])
    #t.vizualize2d(1000)
    ##a.vizualize2d()
    ##a.train()



    #predict=(t.pos_word_by_voc(['съеште', 'школа','господина','приехал',
    #                       'глокая','куздра','штеко','будланула','бокра','и','кудрячит','бокрёнка']))

    X_predict=['съеште', 'школа', 'господина', 'приехал',
     'глокая', 'куздра', 'штеко', 'будланула', 'бокра', 'и', 'кудрячит', 'бокрёнка']
    #X_predict=['символ']
    y_predict=t.pos_word_by_ml(X_predict)

    print(['%s/%s' % (X_predict[i],dg.get(y_predict[i])) for i in range(0,len(y_predict))])

main()