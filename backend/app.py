from flask import Flask, request, jsonify, send_file, render_template
from flask_sqlalchemy import SQLAlchemy
from config import Config
from werkzeug.utils import secure_filename
from flask_cors import CORS
import cv2
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

@app.route('/')
def login():
    return "Log In Page"

@app.route('/upload', methods=['POST'])
def upload_video():
    user_name = request.form['user_name']
    file_name = request.form['file_name']
    video = request.files['video']

    # Ensure the file name ends with .mp4
    if not file_name.endswith('.mp4'):
        file_name += '.mp4'

    # Secure the filename and create a folder based on the file name (without extension)
    secure_name = secure_filename(file_name)
    folder_name = os.path.splitext(secure_name)[0]
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Save video to the folder
    video_path = os.path.join(folder_path, secure_name)
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

    extract_frames(video_path, folder_path)

    return jsonify({"message": "Video uploaded successfully!"}), 201

def extract_frames(video_path, output_folder, frame_rate=24):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)
    
    count = 0
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save only the frames based on the specified frame rate
        if count % interval == 0:
            frame_filename = f"frame_{frame_number:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_number += 1

        count += 1

    cap.release()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4001)
