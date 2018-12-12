import xml.etree.ElementTree as ET
import pandas as pd
import io
from common import Environment
from timeit import default_timer as timer

class OpenCorpus:

    #Read corpus XML dictionary and writes to CSV
    def dict_xml2csv(self, persistent=True, lines=10000):
        t_start = timer()
        env = Environment()
        dfgram = self.grammemes()
        filename_dict = env.filename_dict_xml()
        dfcols = ['word', 'gram', 'idgram']
        df_xml = pd.DataFrame(columns=dfcols)
        env.debug(1, ['CORPUS','Start to load dictionary from XML:', filename_dict])
        try:
            fp = io.open(filename_dict, mode="r", encoding="utf-8")
        except:
            env.debug(1,['CORPUS','Failed to open dictionary file XML:', filename_dict])
        else:
            number_lines = sum(1 for line in fp)
            fp.seek(0)
            t_end = timer()
            env.debug(1, ['CORPUS','File opened:','lines','%s' % number_lines, 'time:', env.job_time(t_start, t_end)])
            t_start = timer()
            step = number_lines // lines
            env.debug(1,['CORPUS','Read dictionary:', filename_dict,'lines: %s step %s' % (lines,step)])
            n_line = 0
            for i in range (0,number_lines):
                line=fp.readline()
                #print(line[5:10])
                if (line[5:10]=='lemma') and (n_line==0):
                    #print(line)
                    tree = ET.fromstring(line)
                    for elem in tree.iter('l'):
                        s_word = elem.attrib.get('t')
                        gram=['',0]
                        j=0
                        for elem2 in elem.iter('g'):
                            gram[j] = elem2.attrib.get('v')
                            break;
                        gram[1] = int(dfgram.index[dfgram['name'] == gram[0]].tolist()[0])
                    #print(s_word,gram)
                    s = pd.Series(data=[s_word, gram[0], gram[1]],index=dfcols)
                    df_xml = df_xml.append(s, ignore_index=True)
                    n_line+=1
                n_line+=1
                if n_line>=step:
                    n_line = 0
            fp.close()
            df_xml.index.name = 'idcorpus'
            t_end = timer()
            env.debug(1, ['CORPUS', 'Dictionary loaded:', 'time:', env.job_time(t_start, t_end)])
            if persistent:
                filename_csv = env.filename_dict_csv()
                env.debug(1,['CORPUS','Write dictionary to CSV:', filename_csv])
                df_xml.to_csv(filename_csv, encoding='utf-8')
                env.debug(1, ['CORPUS', 'Dictionary saved:', filename_csv])
        return df_xml

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
    def grammemes(self, mode=0):
        env = Environment()
        dfgram = pd.DataFrame()
        filename_gram = env.filename_grammemes_csv()
        try:
            dfgram = pd.read_csv(filename_gram, index_col='idgram', encoding='utf-8')
        except:
            env.debug(1, ['Failed to load grammemes CSV file', filename_gram])
        else:
            env.debug(1, ['Load grammemes CSV file', filename_gram])
        if mode==1:
            return dfgram.to_dict().get('name')
        else:
            return dfgram

    #Read corpus XML files and writes to CSV
    def corpus_xml2csv(self, num=1, persistent=True):
        env = Environment()
        file_xml=env.filename_corpus_xml(num)
        df_xml = pd.DataFrame()
        df_gram = self.grammemes()
        dgram = df_gram.to_dict().get('name')
        try:
            tree = ET.ElementTree(file=file_xml)
        except:
            env.debug(1, ['Failed to load XML:', file_xml])
        else:
            t_start = timer()
            env.debug(1, ['CORPUS','XML to CSV:', file_xml])
            for elem in tree.iter('token'):
                #print(elem.tag, elem.attrib)
                serie = pd.Series(data=[])
                badd = False
                s_text = elem.attrib.get('text')
                serie[len(serie)] = s_text.lower()
                for elem2 in elem.iter('g'):
                    #print(elem2.tag, elem2.attrib)
                    sgram = elem2.attrib.get('v')
                    sgram = sgram.upper()
                    if ( df_gram[df_gram['name'].isin([sgram]) == True].size ) > 0 :
                        serie[len(serie)] = sgram
                        serie[len(serie)] = int(df_gram.index[df_gram['name'] == sgram].tolist()[0])
                        #serie[len(serie)] = list(dgram.keys())[list(dgram.values()).index(sgram)]
                        badd = True
                    break
                #print(s)
                if badd:
                    df_xml = df_xml.append(serie, ignore_index=True)
            if not df_xml.empty:
                df_xml = df_xml.drop_duplicates()
                df_xml = df_xml.reset_index(drop=True)
                df_xml.index.name = 'idcorpus'
                df_xml.columns = ['word', 'gram', 'idgram']
                df_xml = df_xml.astype({"idgram": int})
                if persistent:
                    file_csv = env.filename_corpus_csv(num)
                    env.debug(1, ['Write corpus file to CSV:', file_csv])
                    df_xml.to_csv(file_csv, encoding='utf-8')
                    t_end = timer()
                    env.debug(1, ['CORPUS', 'CSV written:', file_csv, 'takes %s sec.' % env.job_time(t_start, t_end)])
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
        #dfgram = self.grammemes()
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
        df_voc.columns = ['word', 'gram','idgram']
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
        file_voc = env.filename_vocabulary_csv() #from vocabulary file
        file_dict = env.filename_dict_csv() #from dictionary file
        try:
            df_voc = pd.read_csv(file_voc, index_col='idcorpus', encoding='utf-8')
        except:
            env.debug(1, ['Failed to read vocabulary file:', file_voc])
        else:
            env.debug(1, ['Read vocabulary OK:', file_voc])
        try:
            df_dict = pd.read_csv(file_dict, index_col='idcorpus', encoding='utf-8')
        except:
            env.debug(1, ['Failed to read dictionary file:', file_dict])
        else:
            env.debug(1, ['Read dictionary OK:', file_dict])
        #Concat
        df_res = pd.concat([df_voc, df_dict])
        df_res = df_res.drop_duplicates()
        #Apply patch words
        df_patch = pd.read_csv(env.filename_vocabulary_patch_csv(), index_col='idcorpus', encoding='utf-8')
        df_res = df_res.drop(df_res[df_res['word'].isin(df_patch['word'])].index, axis=0)
        df_res = pd.concat([df_res, df_patch])
        #print(df_res[df_res['word'].isin(df_patch['word'])])
        df_res = df_res.reset_index(drop=True)
        df_res.index.name = 'idcorpus'
        #print(df_res)
        return df_res

