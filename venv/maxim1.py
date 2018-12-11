from corpus import OpenCorpus
from postagger import POSTagger
from analyzer import mlAnalyzer

c = OpenCorpus()
#data = c.vocabulary()
#print(data.head(1000))

p = POSTagger()
#voc = c.vocabulary()
#p.tokenize(voc)
#p.train(n_splits=5)

m = mlAnalyzer()
#m.process_from_texts_file([21])
m.vizualize_results_pca()


#print(p.pos_word_by_voc(['съеште', 'школа','господина','приехал',
#                         'глокая','куздра','штеко','будланула','бокра','и','кудрячит','бокрёнка']))
#print(p.pos_word_by_ml(['съеште','школа','господина','приехал',
#                        'глокая','куздра','штеко','будланула','бокра','и','кудрячит','бокрёнка']))