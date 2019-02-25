from flask import  Flask
from flask import render_template
from flask import request
app = Flask(__name__)

import sys
sys.path.append("/home/ayushpatidar/PycharmProjects/Automatic_predictive_modeling/")
from loadfile import main_function

@app.route("/", methods = ['POST', 'GET'])
def home():
    return render_template("login.html")

#
@app.route("/result",methods = ['POST', 'GET'])
def result():
    if request.method == "POST":
        result = request.form
        print(result)
        print("FILE PATH IS", result["FILE_PATH"])
        main_function(result["FILE_PATH"], result["TARGET_NAME"], result["PROBLEM_TYPE"])

        return ("training_started")






@app.route("/salvador")
def home1():
    return "here i am "

@app.route("/about")
def here():
    return  render_template("about.html")


if __name__ == "__main__":
    app.run(debug = True)
