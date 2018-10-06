"""DATABASE CONNECTIVITYY FILE..."""

import MySQLdb

try:
    db = MySQLdb.connect("localhost", "root", "ayushpatidar@04", "TIME_SERIES")
    print("DB conencted")
except Exception as e:
    print("DB not connected,{}".format(e))

cur = db.cursor()


try:
    sql = "DROP TABLE IF EXISTS RESULTS"
    cur.execute(sql)

except Exception as e:
    print("error while dropping table,{}".format(e))

try:

    sql = """CREATE TABLE RESULTS(TRAINING_ID INT NOT NULL AUTO_INCREMENT,
                                  MODEL_NAME VARCHAR(100) NULL,
                                  MODEL_ACCURACY FLOAT NULL,
                                  TRAINING_ERROR VARCHAR(550) NULL,
                                  PRIMARY KEY(TRAINING_ID)
                                  )"""

    cur.execute(sql)
    print("table created")

    return cur



except Exception as e:
    print("error while creating table,{}".format(e))