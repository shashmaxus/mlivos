import pandas as pd
import numpy as np
import pickle
import time
import codecs

from jinja2 import Environment as jinjaEnvironment, FileSystemLoader, Template

from common import Environment
from corpus import OpenCorpus
from analyzer import mlAnalyzer

class Reporter:
    def make_report(self):
        env = Environment()
        a = mlAnalyzer()
        #template = Template('Hello {{ name }}!')
        #print(template.render(name=u'Вася'))
        jenv = jinjaEnvironment(loader = FileSystemLoader(env.path_templates()))
        #print(jenv.loader)
        template = jenv.get_template("report_global.tpl.html")

        data = a.get_texts_stat() #Статистика по текстам из файла обучающей выборки
        test = a.get_texts_stat(mode='test') #Статистика по текстам тестовой выборки
        #print(data)
        #test['predict'] = test['predict'].astype(int)
        test['validation'] = 0
        test.loc[test.idauthor == test.predict,'validation'] = 1
        print(data)
        print(test)

        #Summary stat
        group = pd.merge(data, test, on='idauthor', how='left', suffixes=('', '_test'))
        print(group)
        group = group.groupby(['idauthor', 'name_author'], as_index=False).agg({'idtext' : ['nunique'],
                                                                'words_chunk' : ['sum'],
                                                                'name_test': ['nunique'],
                                                                'words_chunk_test': ['sum'],
                                                                'validation' : ['mean']
                                                                    })


        print(group)
        group.drop(['idauthor'], axis=1, inplace=True)
        group.sort_values('name_author', inplace=True)
        #Переименовать колонки в русскоязычные
        group.columns = ['Писатель',
                         'Кол-во текстов для обучения',
                         'Объём текстов для обучения (кол-во слов)',
                         'Кол-во текстов для проверки',
                         'Объём текстов для проверки (кол-во слов)',
                         'Точность определения'
                         ]
        n_accuracy = group['Точность определения'].mean()
        #Целые числа показываем без дробной части
        int_cols = ['Кол-во текстов для обучения',
                         'Объём текстов для обучения (кол-во слов)',
                         'Кол-во текстов для проверки',
                         'Объём текстов для проверки (кол-во слов)']
        for col in int_cols:
            group[col] = group[col].astype(int)
        group.reset_index(drop = True, inplace = True)
        s = group.style.set_properties(**{'text-align': 'right'})
        group.fillna('', inplace = True)
        s.hide_index().render()

        #Training stat
        group_train =  data.groupby(['author'], as_index=False).agg({'idauthor' : ['count'],
                                                                'sentences_text' : ['sum'],
                                                                'words_text' : ['sum'],
                                                                'sentence_mean': ['mean'],
                                                                'name': [lambda col: '<br />'.join(col)],
                                                                    })
        group_train.reset_index(drop = True, inplace = True)
        s_train = group_train.style.set_properties(**{'text-align': 'right'})
        group_train.fillna('', inplace=True)
        group_train.columns = ['Писатель',
                               'Кол-во текстов',
                               'Кол-во предложений',
                               'Кол-во слов',
                               'Средняя длина предложения',
                               'Произведения'
                         ]
        n_train = group_train['Кол-во текстов'].sum()
        s_train.hide_index().render()

        # Testing stat
        group_test = test.groupby(['author'], as_index=False).agg({'idauthor': ['count'],
                                                                    'sentences_text': ['sum'],
                                                                    'words_text': ['sum'],
                                                                    'sentence_mean': ['mean'],
                                                                    'name': [lambda col: '<br />'.join(col)],
                                                                    'validation': ['mean'],
                                                                    'shortname_predict': [lambda col: '<br />'.join(col)],
                                                                    })
        group_test.reset_index(drop=True, inplace=True)
        s_test = group_test.style.set_properties(**{'text-align': 'right'})
        group_test.fillna('', inplace=True)
        group_test.columns = ['Писатель',
                               'Кол-во текстов',
                               'Кол-во предложений',
                               'Кол-во слов',
                               'Средняя длина предложения',
                               'Произведения',
                               'Результат проверки',
                               'Определён автор',
                               ]
        n_test = group_test['Кол-во текстов'].sum()
        s_test.hide_index().render()

        template_vars = {"title": "Отчёт",
                         "detection_accuracy": '%s' % (round(n_accuracy,4)*100),
                         "train_texts_pivot_table_style_render" : s.render(),
                         "n_train_texts": round(n_train,0),
                         "train_texts_table_style_render": s_train.render(),
                         "n_test_texts": round(n_test,0),
                         "test_texts_table_style_render": s_test.render()
                         }
        html_out = template.render(template_vars)

        file = codecs.open(env.filename_global_report_html(), "w", "utf-8-sig")
        file.write(html_out)
        file.close()
        #print(html_out)
        return html_out