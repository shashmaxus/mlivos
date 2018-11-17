import xml.etree.ElementTree as ET
import pandas as pd
from common import Environment

class OpenCorpus:

    #Read corpus XML grammemes and writes to CSV
    def grammemes_xml2csv(self, persistent=True):
        env = Environment()
        filename_gram = env.filename_grammemes_xml()
        dfcols = ['name', 'alias', 'description']
        df_xml = pd.DataFrame(columns=dfcols)
        try:
            tree = ET.ElementTree(file=filename_gram)
        except:
            env.debug(1,['Failed to load grammemes from XML:', filename_gram])
        else:
            env.debug(1,['Read grammemes:', filename_gram])
            for elem in tree.iter('grammeme'):
                #print(elem.tag, elem.attrib)
                sattr = elem.attrib.get('include')
                if sattr == 'on':
                    sname = sali = sdesc = ''
                    for child in elem:
                        if child.tag.lower() == 'name':
                            sname = child.text.upper()
                        elif child.tag.lower() == 'alias':
                            sali = child.text.upper()
                        elif child.tag.lower() == 'description':
                            sdesc = child.text.lower()
                    s = pd.Series(data=[sname, sali, sdesc],index=dfcols)
                    df_xml = df_xml.append(s, ignore_index=True)
            df_xml.index.name = 'idgram'
            if persistent:
                filename_csv = env.filename_grammemes_csv()
                env.debug(1,['Write grammemes to CSV:', filename_csv])
                df_xml.to_csv(filename_csv, encoding='utf-8')
        return df_xml

    #Return grammemes dataset
    def grammemes(self):
        env = Environment()
        dfgram = pd.DataFrame()
        filename_gram = env.filename_grammemes_csv()
        try:
            dfgram = pd.read_csv(filename_gram, index_col='idgram', encoding='utf-8')
        except:
            env.debug(1, ['Failed to load grammemes CSV file', filename_gram])
        else:
            env.debug(1, ['Load grammemes CSV file', filename_gram])
        return dfgram

    #Read corpus XML files and writes to CSV
    def corpus_xml2csv(self, num=1, persistent=True):
        env = Environment()
        file_xml=env.filename_corpus_xml(num)
        df_xml = pd.DataFrame()
        df_gram = self.grammemes()
        try:
            tree = ET.ElementTree(file=file_xml)
        except:
            env.debug(1, ['Failed to load XML:', file_xml])
        else:
            for elem in tree.iter('token'):
                #print(elem.tag, elem.attrib)
                serie = pd.Series([])
                badd = False
                s_text = elem.attrib.get('text')
                serie[len(serie)] = s_text.lower()
                for elem2 in elem.iter('g'):
                    #print(elem2.tag, elem2.attrib)
                    sgram = elem2.attrib.get('v')
                    sgram = sgram.upper()
                    if ( df_gram[df_gram['name'].isin([sgram]) == True].size ) > 0 :
                        serie[len(serie)] = sgram
                        badd = True
                    break
                #print(s)
                if badd:
                    df_xml = df_xml.append(serie, ignore_index=True)
            df_xml = df_xml.drop_duplicates()
            df_xml = df_xml.reset_index(drop=True)
            df_xml.index.name = 'idcorpus'
            df_xml.columns = ['word', 'gram']
            if persistent:
                file_csv = env.filename_corpus_csv(num)
                env.debug(1, ['Write corpus file to CSV:', file_csv])
                df_xml.to_csv(file_csv, encoding='utf-8')
        return df_xml

    # Read corpus XML files and writes text to TXT
    def corpus_xml2txt(self, num=1, persistent=True):
        result = True
        env = Environment()
        file_xml = env.filename_corpus_xml(num)
        try:
            tree = ET.ElementTree(file=file_xml)
        except:
            env.debug(1, ['Failed to load XML:', file_xml])
            result = False
        else:
            file_txt = env.filename_corpus_txt(num)
            file = open(file_txt, mode='w')
            for elem in tree.iter('source'):
                # print(elem.text, elem.tag, elem.attrib)
                file.write(elem.text)
                file.write(' ')
            file.close()
            env.debug(1, ['Write corpus file to TXT:', file_txt])
        return result

    #Create vocabulary from cospus files
    def vocabulary_from_corpus(self, n_min=1, n_max=10, persistent=True):
        env = Environment()
        df_voc = pd.DataFrame()
        for i in range(n_min, n_max+1):
            file_csv = env.filename_corpus_csv(i)
            try:
                dffile = pd.read_csv(file_csv, index_col='idcorpus', encoding='utf-8')
            except:
                env.debug(1, ['Failed to read corpus file:', file_csv])
            else:
                env.debug(1, ['Read OK:', file_csv])
                if not dffile.empty:
                    df_voc = df_voc.append(dffile)
        df_voc = df_voc.drop_duplicates()
        df_voc.columns = ['word', 'gram']
        df_voc = df_voc.reset_index(drop=True)
        df_voc.index.name = 'idcorpus'
        if persistent:
            file_voc=env.filename_vocabulary_csv()
            env.debug(1, ['Write vocabulary to CSV:', file_voc])
            df_voc.to_csv(file_voc, encoding='utf-8')
        return df_voc


    # Return vocabulary dataset
    def vocabulary(self):
        env = Environment()
        df_voc = pd.DataFrame()
        file_voc = env.filename_vocabulary_csv()
        try:
            df_voc = pd.read_csv(file_voc, index_col='idcorpus', encoding='utf-8')
        except:
            env.debug(1, ['Failed to read vocabulary file:', file_voc])
        else:
            env.debug(1, ['Read vocabulary OK:', file_voc])
        return df_voc

