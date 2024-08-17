from flask import Flask, request, jsonify, send_file, render_template
from flask_sqlalchemy import SQLAlchemy
from config import Config
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)
db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)

class VideoUpload(db.Model):
    __tablename__ = 'video_uploads'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    file_name = db.Column(db.String(255), nullable=False)
    video_path = db.Column(db.Text, nullable=False)
    uploaded_at = db.Column(db.DateTime, default=db.func.current_timestamp())

@app.route('/upload', methods=['POST'])
def upload_video():
    user_name = request.form['user_name']
    file_name = request.form['file_name']
    video = request.files['video']

    # Ensure the file name ends with .mp4
    if not file_name.endswith('.mp4'):
        file_name += '.mp4'
    # Save video to filesystem
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    video.save(video_path)

    # Insert into database
    user = User.query.filter_by(name=user_name).first()
    if not user:
        user = User(name=user_name)
        db.session.add(user)
        db.session.commit()

    video_upload = VideoUpload(user_id=user.id, file_name=file_name, video_path=video_path)
    db.session.add(video_upload)
    db.session.commit()

    return jsonify({"message": "Video uploaded successfully!"}), 201

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
    app.run(debug=True, host="0.0.0.0", port=4001)
