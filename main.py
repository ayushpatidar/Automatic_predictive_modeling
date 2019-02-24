from flask import  Flask
from flask import render_template
from flask import request
app = Flask(__name__)

import sys
sys.path.append("/home/ayushpatidar/PycharmProjects/Automatic_predictive_modeling/")
from loadfile import main_function
import MySQLdb
from mysqlclient import user_authentication
from mysqlclient import create_user
from mysqlclient import create_user_table


@app.route('/register', methods = ["POST", "GET"])
def register_user():
    print("in register function")

    if request.method == "POST":

        create_user_table()

        create_user(request.form["auth_user"], request.form["auth_pass"],
                    request.form["first_name"], request.form["last_name"],
                    request.form["city"])



        return ("REGISTERED PLEASE GOT TO LOGIN PAGE")




    return  render_template("register.html", user_image = "static/images/logo.png")



@app.route('/login', methods= ['POST','GET'])
def login():
    if request.method == "POST":
        r = user_authentication(request.form["auth_user"], request.form["auth_pass"])
        if r ==1:
            return render_template("login.html")
        else:
            return ("INVALID USERNAME OR PASSWORD")

    return render_template("faltu.html", user_image = "static/images/logo.png")



@app.route("/", methods = ['POST'])
def home():
    return render_template("login.html")

#
@app.route("/result",methods = ['POST'])
def result():
    if request.method == "POST":
        result = request.form
        print(result)
        print("FILE PATH IS", result["FILE_PATH"])
        df = main_function(result["FILE_PATH"], result["TARGET_NAME"], result["PROBLEM_TYPE"])

        return df.to_html(header="true", table_id="table")






if __name__ == "__main__":
    app.run(debug = True)
