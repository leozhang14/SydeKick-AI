from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Malhadinho is king. #FakhretdinovAndNewInshallah"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3001)
