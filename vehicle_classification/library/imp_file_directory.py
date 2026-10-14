import os
import copy

def search_last_leaves(path):
    resL = []
    for dirname, dirnames, filenames in os.walk(path):
        #print('dirname -->', dirname)
        #print('dirnames -->', dirnames)        
        if dirnames == []:            
            dic = {}
            dic['path'] = dirname
            resL.append(dic)
    return resL

def search_every_files(resL, search):
    dic = {}
    number = 1
    for e in resL:
        dic[e['path']]=[]
        for filename in os.listdir(e['path']):
            if search in filename:
                dic[e['path']].append({'name':filename,'number':number})
                number += 1
    return dic, number-1

