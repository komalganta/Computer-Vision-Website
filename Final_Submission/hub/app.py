from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # This will load hub/templates/landing.html
    return render_template('landing.html')