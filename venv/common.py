import numpy as np
import pandas as pd
import nltk
from itertools import zip_longest

class Environment:

    rus_letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    n_bgm_features = 0
    a_bgm_top = []
    a_bgm_rare = []
    a_bgm_excluded = []
    di_bgm_byletters = {}
    di_bgm_byfeatures = {}
    analyzer_max_words = 1000000 #Максимальное число слов в документе, которые могут входить в одну часть (chunk)

    def __init__(self):
        if len(Environment.a_bgm_top)==0:
            Environment.configure_nltk_path()
            Environment.di_bgm_byletters, Environment.di_bgm_byfeatures = Environment.bgm_di_letters() #max features
            Environment.n_bgm_features, Environment.a_bgm_top, Environment.a_bgm_rare = Environment.bgm_stat(n_penalty_top=0.05, n_penalty_rare=0.05)
            Environment.a_bgm_excluded.extend(Environment.a_bgm_top)
            Environment.a_bgm_excluded.extend(Environment.a_bgm_rare)
            Environment.di_bgm_byletters, Environment.di_bgm_byfeatures = Environment.bgm_di_letters()
        #, arare_bigrams = self.bgm_stat()

    def debug_info_level(self):
        return 1

    def filename_mlcache_csv(self):
        return "c:/prj/mlivos_data/db/cache_ml.csv"

    def filename_dict_xml(self):
        return "C:/Prj/corpus/dict.opcorpora.xml"

    def filename_dict_csv(self):
        return "C:/Prj/mlivos_data/db/dict.csv"

    def filename_grammemes_xml(self):
        return "c:/prj/corpus/grammemes.xml"

    def filename_grammemes_csv(self):
        return "c:/prj/mlivos_data/db/grammemes.csv"

    def filename_vocabulary_csv(self):
        return "c:/prj/mlivos_data/db/vocabulary.csv"

    def filename_vocabulary_patch_csv(self):
        return "c:/prj/mlivos_data/db/vocabulary_patch.csv"

    def filename_tokenz_csv(self):
        return "c:/prj/mlivos_data/db/tokenz.csv"

    def filename_model_tree(self):
        return "c:/prj/mlivos_data/model/posmodel.tree"

    def filename_model_texts(self):
        return "c:/prj/mlivos_data/model/model.texts"

    def filename_model_texts_pca(self):
        return "c:/prj/mlivos_data/model/model.texts.pca"

    def filename_scaler(self):
        return "c:/prj/mlivos_data/model/scaler"

    def filename_results_csv(self):
        return "c:/prj/mlivos_data/db/results.csv"

    def filename_stat_test_csv(self):
        return "c:/prj/mlivos_data/db/stat_test.csv"

    def filename_authors_csv(self):
        return "c:/prj/mlivos_data/db/authors.csv"

    def filename_texts_csv(self):
        return "c:/prj/mlivos_data/db/texts_train.csv"

    def filename_predict_csv(self):
        return "c:/prj/mlivos_data/db/texts_test.csv"

    def filename_corpus_xml(self, num):
        return "c:/prj/corpus/annot/%d.xml" % (num,)

    def filename_corpus_csv(self, num):
        return "c:/prj/corpus/csv/%d.csv" % (num,)

    def filename_corpus_txt(self, num):
        return "c:/prj/corpus/txt/%d.txt" % (num,)

    def filename_xtrain_csv(self):
        return "c:/prj/mlivos_data/temp/pos_train_X.csv"

    def filename_test_err_csv(self):
        return "c:/prj/mlivos_data/temp/test_err.csv"

    def filename_stat_bigram_letters_csv(self):
        return "c:/prj/mlivos_data/db/stat_bigram_letters.csv"

    def filename_stat_pos_tokenz_csv(self):
        return "c:/prj/mlivos_data/temp/stat_pos_tokenz.csv"

    def path_templates(self):
        return "c:/prj/mlivos_data/templates/"

    def filename_global_report_html(self):
        return "c:/prj/mlivos_data/reports/global_report.html"

    def cache_ml_csv(self):
        return "c:/prj/mlivos_data/db/ml_cache.csv"

    @staticmethod
    def configure_nltk_path():
        nltk.data.path.append('c:/app/nltk_data')
        return 0

    def list_rus_letters(self):
        return Environment.rus_letters

    def debug(self, severity, ainfo):
        if severity <= self.debug_info_level():
            s = ''
            for info in ainfo:
                s += info
                s += ' '
            print(s)
        return True

    def job_time(self, t_start, t_end):
        return str(round((t_end-t_start),2))

    @staticmethod
    def bgm_stat(n_penalty_top=0.1, n_penalty_rare=0.1):
        filename_stat_bigram_letters_csv = "c:/prj/mlivos_data/temp/stat_bigram_letters.csv"
        n_top = 0
        a_top = []
        n_rare = 0
        a_rare = []
        n_count = len(Environment.di_bgm_byletters)+1
        try:
            df_bgm_stat = pd.read_csv(filename_stat_bigram_letters_csv, index_col='idbigram', encoding='utf-8')
        except:
            #debug(1, ['Failed to read bigrams stat file:', filename_stat_bigram_letters_csv()])
            print('Failed to read bigrams stat file:', filename_stat_bigram_letters_csv)
        else:
            #debug(1, ['Read bigrams stat OK:', filename_stat_bigram_letters_csv()])
            print('Read bigrams stat OK:', filename_stat_bigram_letters_csv)
            n_sum = df_bgm_stat['counts'].sum()
            n_count = df_bgm_stat.shape[0]
            for index, serie in df_bgm_stat.iterrows():
                n_top+=df_bgm_stat.at[index, 'counts']
                if (n_top/n_sum) < n_penalty_top:
                    a_top.append(df_bgm_stat.at[index,'bigram'])
                else:
                    break
            df_bgm_stat = df_bgm_stat.sort_values(by=['counts'], ascending=True)
            #print(df_bgm_stat)
            for index, serie in df_bgm_stat.iterrows():
                n_rare+=df_bgm_stat.at[index, 'counts']
                if (n_rare/n_sum) < n_penalty_rare:
                    a_rare.append(df_bgm_stat.at[index,'bigram'])
                else:
                    break
            #a_rare = list(df_bgm_stat[df_bgm_stat['counts'] <= 10]['bigram'])
            n_count = n_count - len(a_top) - len(a_rare)
            print('Bigrams',n_count,':\nTop',len(a_top),':',a_top,'\nRare',len(a_rare),':',a_rare,'counts_sum: %s' % n_sum, 'features', n_count)
        return n_count, a_top, a_rare

    @staticmethod
    def bgm_di_letters():
        di_letters = {}
        di_features = {}
        a_letter = list(Environment.rus_letters)
        bgm_columns_i = Environment.bgm_columns_list(mode=0)
        n_letters = 0
        print(Environment.a_bgm_excluded)
        for l1 in a_letter:
            for l2 in a_letter:
                s_pair = '%s%s' % (l1,l2)
                if s_pair not in Environment.a_bgm_excluded:
                    n_letters+=1
                    di_letters[s_pair] = n_letters
                    di_features[n_letters] = s_pair
        print('BGM dictionaries ready',len(di_letters),':\n',di_letters,'\n',di_features)
        return (di_letters, di_features)

    #Bgm columns list
    @staticmethod
    def bgm_columns_list(mode=1):
        n_f = Environment.n_bgm_features+1
        if mode==0:
            bgm_columns = [(i) for i in range(0, n_f, 1)]
        if mode==1:
            bgm_columns = ['bgm_l_%s' % (i) for i in range(0, n_f, 1)]
        return bgm_columns

    @staticmethod
    # Create a function called "chunks" with two arguments, l and n:
    def chunks(l, n):
        for i in range(0, n):
            yield l[i::n]

    @staticmethod
    def model_performance_plot(predictions, targets, title):
        # Get min and max values of the predictions and targets.
        min_val = max(max(predictions), max(targets))
        max_val = min(min(predictions), min(targets))
        # Create dataframe with predicitons and targets.
        performance_df = pd.DataFrame({"target": targets})
        performance_df["predict"] = predictions
        print(performance_df.head())

        # Plot data
        sns.jointplot(y="target", x="predict", data=performance_df, kind="reg", height=7)
        plt.plot([min_val, max_val], [min_val, max_val], 'm--')
        plt.title(title, fontsize=9)
        plt.show()

    @staticmethod
    def downcast_dtypes(df):
        float_cols = [c for c in df if df[c].dtype == "float64"]
        int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
        df[float_cols] = df[float_cols].astype(np.float32)
        df[int_cols] = df[int_cols].astype(np.int16)
        return df