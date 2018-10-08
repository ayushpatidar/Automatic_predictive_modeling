"""DATABASE CONNECTIVITYY FILE..."""
import MySQLdb
import os
import warnings
import sys

warnings.filterwarnings('ignore')


def connection():
    

    db = MySQLdb.connect("localhost", "root", "ayushpatidar@04", "TIME_SERIES")
    print("DB conencted")



    cur = db.cursor()

    try:
        f = 0
        sql = """SHOW TABLES"""
        cur.execute(sql)
        rs = cur.fetchall()
        rs = [item[0] for item in rs]
        if "RESULTS" in rs:
            f = 1
    except Exception as e:
        print("error while fetching table,{}".format(e))
        exit()

    try:
        if f==0:
            sql = """CREATE TABLE RESULTS(TRAINING_ID INT NOT NULL AUTO_INCREMENT,
                                          MODEL_NAME VARCHAR(100) NULL,
                                          MODEL_ACCURACY FLOAT NULL,
                                          TRAINING_ERROR VARCHAR(550) NULL,
                                          PRIMARY KEY(TRAINING_ID)
                                          )"""

            cur.execute(sql)
            print("table created")
            #db.close()
            return (db)
        else:
            print("TABLE IS ALREADY THERE")
            return (db)




    except Exception as e:
        db.close()
        print("error while creating table,{}".format(e))
