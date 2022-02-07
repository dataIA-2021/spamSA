

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:50:49 2022

@author: alexiadeboynes
"""
# import des librairies
import sqlite3
#import mysql.connector
import pymysql
import psycopg2
from sqlalchemy import create_engine
import os

import yaml
import pandas as pd


import pandas as pd
import re

#loading data

df= pd.read_csv("spam.csv", encoding='latin-1')
#data.info()

# Dropping the 3 unnamed  collumns

to_drop = ["Unnamed: 2","Unnamed: 3","Unnamed: 4"]
df = df.drop(df[to_drop], axis=1)


# Renaming the columns
df.rename(columns = {"v1":"label", "v2":"message"}, inplace = True)
#df.head()


#fonction

def countspam(code):
    n=0
    countsp = 0
    countha = 0
    for line in df['message']: # Parcours des lignes
        x = re.findall(code, line)
        if x:
            print(n, 'findall:', x)#commenter les prints pour de la visibilité
            print(df['label'][n])
            print(df['message'][n])
        
            if df['label'][n]== 'spam':
                countsp +=1
            else:
                countha +=1     
        n+=1
    print('Nombre de Spam :', countsp)
    print('Nombre de Ham :', countha)
    
countspam("prize")