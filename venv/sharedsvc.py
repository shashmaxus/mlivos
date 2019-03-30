import pandas as pd
import numpy as np
import zlib
from timeit import default_timer as timer

from common import Environment

class Word_Encoder:
    def s_encode(self, s):
        #return (zlib.adler32(str.encode(s))*pow(10,-10))
        value = float(zlib.adler32(str.encode(s)))
        while (value > 1):
            value = value / 10
        return value

    def s2token(self, index, serie):
        a_result = serie.values
        s_word=a_result[0]
        a_result[6] = self.s_encode(s_word) #token
        a_result[7] = len(s_word) #len
        a_result[2] = s_word[-2:] #s2
        a_result[8] = self.s_encode(s_word[-2:]) #ts2
        a_result[3] = s_word[-3:]  # s3
        a_result[9] = self.s_encode(s_word[-3:])  # ts3
        a_result[4] = s_word[:2]  # p2
        a_result[10] = self.s_encode(s_word[:2])  # tp2
        a_result[5] = s_word[:3]  # p3
        a_result[11] = self.s_encode(s_word[:3])  # tp3
        return a_result

    def word2token(self, s):
        t_start = timer()
        env = Environment()
        bgm_columns = env.bgm_columns_list(mode=1)
        n_shift = 5

        a_result=np.zeros(len(bgm_columns)+n_shift)
        a_result[0] = len(s)
        a_result[1] = self.s_encode(s[-2:])  # ts2
        a_result[2] = self.s_encode(s[-3:])  # ts3
        a_result[3] = self.s_encode(s[2:])  # tp2
        a_result[4] = self.s_encode(s[3:])  # tp3

        t_end = timer()
        #env.debug(1, ['WordEncoder', 'word2token', '%s without bgm takes %s sec.' % (s, env.job_time(t_start, t_end))])
        #t_start = timer()

        di_letters = env.di_bgm_byletters
        #print(di_letters)
        di_word = {}
        for n_l in range(0, len(s) - 1):
            n_l2 = n_l + 1
            di_n = di_letters.get('%s%s' % (s[n_l], s[n_l2]))
            #print('%s%s' % (s[n_l], s[n_l2]),di_n)
            if di_n is not None:
                #print(di_n)
                a_result[di_n + n_shift] = 1
        t_end = timer()
        #env.debug(1, ['WordEncoder', 'word2token', '%s takes %s sec.' % (s, env.job_time(t_start, t_end))])
        return a_result