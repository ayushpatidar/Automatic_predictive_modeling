"""DATABASE CONNECTIVITYY FILE..."""
import MySQLdb
import os
import warnings
import sys
import pandas as pd

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





def commit_results_db():

    db = MySQLdb.connect("localhost", "root", "ayushpatidar@04", "AUTO_ML")
    print("DB CONNETCED")

    cur = db.cursor()

    try:
        f = 0
        sql = "SHOW TABLES LIKE 'RESULTS'"
        cur.execute(sql)
        rs = cur.fetchone()
        print("rs is ", rs)


        if rs:
            print("table is already there")

        else:
            print("create a table with name RESULTS")

            sql = """CREATE TABLE RESULTS(DATASET_ID VARCHAR(100) NOT NULL, TRAINING_ID VARCHAR(100) NOT NULL, MODEL_NAME VARCHAR(100) NULL,
            ACCURACY FLOAT NULL, FEATURE_SELECTOR VARCHAR(100) NULL, PRIMARY KEY(TRAINING_ID), ERROR_MODEL_TRAINING VARCHAR(100) NULL)"""

            cur.execute(sql)
            db.commit()
            print("table created")


    except Exception as e:
        print("error while creating table ", e)


    db.close()



def set_results_db(data):

     db = MySQLdb.connect("localhost", "root", "ayushpatidar@04", "AUTO_ML")
     print("db connected")

     cur = db.cursor()

     try:
         print("trying to insert in db")
         sql = "INSERT INTO RESULTS(DATASET_ID, TRAINING_ID, MODEL_NAME, ACCURACY, FEATURE_SELECTOR, ERROR_MODEL_TRAINING)" \
               "VALUES(%s, %s, %s, %s, %s, %s)"

         cur.execute(sql, data)
         print("results added successfully in database AUTO_ML")
         db.commit()


     except Exception as e:
         print("error while inserting content in database AUTO_ML", e)


     db.close()


def fetch_results():

    db = MySQLdb.connect("localhost", "root", "ayushpatidar@04", "AUTO_ML")

    try:
        cur = db.cursor()
        sql = "SELECT * FROM RESULTS"
        cur.execute(sql)
        #print("RESULTS ARE", cur.fetchall())

        result  = list(cur.fetchall())

        sql = "SHOW COLUMNS FROM RESULTS"
        cur.execute(sql)
        col = list(cur.fetchall())
        cols = list()
        for i in col:
            cols.append(i[0])




        print("COLUMNS ARE ",cols)
        df = pd.DataFrame(result, columns=cols)

        db.close()

        return df


    except Exception as e:
        print("error while fetching results", e)



def user_authentication(username, password):
    #this function is used to check whether a user is authenticated or not
    print("in authentication function")

    db = MySQLdb.connect("localhost", "root", "ayushpatidar@04", "USER_LOGIN")
    print("USER_DETAILS DB CONNECTED")

    try:
        cur = db.cursor()

        sql = "SELECT USER_ID, PWD FROM DETAILS"
        cur.execute(sql)

        results = list(cur.fetchall())

        print(results)
        print(type(results))

        if (username,password) in results:
            #use exsist
            return 1
        else:
            #user is not registered
            return 0

    except Exception as e:
        print("error in user_authentication function", e)





def create_user(username, password, f_name, l_name, city):

    data = (username, f_name, l_name, password, city)

    db = MySQLdb.connect("localhost", "root", "ayushpatidar@04", "USER_LOGIN")

    print("db connected")

    try:

        cur = db.cursor()
        sql = """INSERT INTO DETAILS(USER_ID, F_NAME  ,
                L_NAME , PWD, CITY)
                VALUES(%s, %s, %s, %s, %s)"""

        cur.execute(sql, data)
        db.commit()
        print("result successfully added")



    except Exception as e:
        print("error in create_user DB ", e)


    db.close()


def create_user_table():
    print("in create user db table")

    db = MySQLdb.connect("localhost", "root", "ayushpatidar@04", "USER_LOGIN")
    print("db connected")

    try:
        cur = db.cursor()
        sql = "SHOW TABLES LIKE 'DETAILS'"
        cur.execute(sql)

        rs = cur.fetchone()

        if rs:
            print("USER TABLE ALREADY THERE")
        else:
            print("MAKE NEW USER TABLE")

            sql =  """CREATE TABLE DETAILS(USER_ID VARCHAR(100) NOT NULL, F_NAME VARCHAR(100) NULL ,
                L_NAME VARCHAR(100) NULL, PWD VARCHAR(100) NOT NULL, CITY VARCHAR(100) NULL,
                PRIMARY KEY(USER_ID))"""
            cur.execute(sql)
            db.commit()


    except Exception as e:
        print("error in create_user_table ", e)

    db.close()