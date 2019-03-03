import pandas as pd
import nltk

from common import Environment
from corpus import OpenCorpus
from postagger import POSTagger
from analyzer import mlAnalyzer
from sharedsvc import Word_Encoder

def main():
    pd.set_option("display.max_columns", 100)
    pd.set_option('display.width', 1000)

    #Служебные классы
    env = Environment()
    c = OpenCorpus()
    t = POSTagger()
    a = mlAnalyzer()
    enc = Word_Encoder()
    g = pd.DataFrame()
    g = c.grammemes()
    dg = c.grammemes(mode = 1) #Справочник Части речи mode = 1 возвращает в виде словаря python
    da = c.authors(mode=1)  #Справочник - авторы

    #Пример обработки текстов из texts_train и добавления статистической информации в results
    #a_texts_train = [1, 16]
    #a_texts_train = [48]
    #for i in a_texts_train:
    #    a.process_from_texts_file([i])

    #Пример визуализации статистической информации из results в 2-мерном пространстве
    #a.vizualize2d()

    #Пример визуализации статистичесокй информации о частях речи
    #t.vizualize2d(n_frac = 0.001)

    #Предсказание автора текста из text_test
    #[0, 1, 2, 3, 4]) #предсказать все тексты - долго
    text2predict = [3]
    y = a.predict(text2predict) #предсказать - указать номер текста
    j=0
    for i in y:
        print('idtext=%s' % text2predict[j], da.get(i))
        j=j+1

main()