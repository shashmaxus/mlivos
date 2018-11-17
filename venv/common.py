class Environment:
    def debug_info_level(self):
        return 1

    def filename_grammemes_xml(self):
        return "c:/prj/corpus/grammemes.xml"

    def filename_grammemes_csv(self):
        return "c:/prj/temp/dfgram.csv"

    def filename_vocabulary_csv(self):
        return "c:/prj/temp/df_voc.csv"

    def filename_tokenz_csv(self):
        return "c:/prj/temp/df_token.csv"

    def filename_model_tree(self):
        return "c:/prj/temp/model.tree"

    def filename_results_csv(self):
        return "c:/prj/temp/dfres.csv"

    def filename_texts_csv(self):
        return "c:/prj/temp/texts.csv"

    def filename_corpus_xml(self, num):
        return "c:/prj/corpus/annot/%d.xml" % (num,)

    def filename_corpus_csv(self, num):
        return "c:/prj/corpus/csv/%d.csv" % (num,)

    def filename_corpus_txt(self, num):
        return "c:/prj/corpus/txt/%d.txt" % (num,)

    def filename_xtrain_csv(self):
        return "c:/prj/temp/xtrain.csv"

    def debug(self, severity, ainfo):
        if severity <= self.debug_info_level():
            s = ''
            for info in ainfo:
                s += info
                s += ' '
            print(s)
        return True