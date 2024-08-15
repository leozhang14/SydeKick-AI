from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def login():
    return "Log In Page"

# Home page route
@app.route('/home')
def home():
    return "Galpao Da Luta"

# About page route
@app.route('/about')
def about():
    return "21-3, 6'3, 241, Bahia, Salvador, Brazil, #7 HW Contender, Jailton \"Malhadinho\" Almeida"

# Contact page route
@app.route('/contact')
def contact():
    return "#FakhretdinovAndNew"

# Another page route
@app.route('/services')
def services():
    return "BIGI BOY!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3001)
