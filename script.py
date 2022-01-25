#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:38:25 2022

@author: alexiadeboynes
"""

import sqlite3
#import mysql.connector
import pymysql
import psycopg2
from sqlalchemy import create_engine

import yaml
import os
import pandas as pd

with open ('config.yaml', 'r') as f:
    conf = yaml.safe_load(f)


my = conf['MYSQL']


str = f"mysql+pymysql://{my['user']}:{my['password']}@{my['host']}:{my['port']}/{my['database']}"
engine = create_engine(str, echo=False)


path = '/Users/alexiadeboynes/Documents/g4brazilian/'

# lecture de tous les csv
def readCSV(name):
    df = pd.read_csv(name, index_col=False, sep= ',', decimal= '.')
    # print(df)
    df.to_sql(name, con=engine, if_exists='append', index=False)


directory = os.fsencode(path)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith('.csv'):
        readCSV(filename)
    tableName = filename[:-4]