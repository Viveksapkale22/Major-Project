# File: modules/routes.py

from flask import render_template, request, redirect, url_for, session, flash, jsonify, Response
import threading
import time
import cv2
from modules.utils import allowed_file, play_alert, boxes_intersect, detect_motion, save_frame_and_get_path
from modules.auth import login_user, register_user, send_reset_email, logout_user
from modules.face_analysis import analyze_gender
from modules.detection import generate_frames

bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

def register_routes(app, model, tracker, users_collection, bcrypt, user_data_store):
    """Registers all app routes."""

    global_state = {
        "restricted_area": None,
        "counting_enabled": False,
        "stop_video_flag": threading.Event(),
        "selected_gender": "both",
        "gender_labels": {},
        "processing": {},
        "last_alert_time": {},
        "person_tracks": {},
        "person_counter": 0,
        "detected_persons": [],
        "email": "viveksapkale022@gmail.com"
    }

    @app.route('/login', methods=['POST'])
    def login():
        return login_user(request, session, users_collection, bcrypt, user_data_store)

    @app.route('/register', methods=['POST'])
    def register():
        return register_user(request, users_collection, bcrypt)

    @app.route('/logout', methods=['POST'])
    def logout():
        return logout_user(session)

    @app.route('/forget-password', methods=['POST'])
    def forget_password():
        email = request.form.get("email")
        return send_reset_email(email, users_collection)

    @app.route('/main')
    def main():
        if 'username' not in session:
            flash('You need to log in first!')
            return redirect(url_for('index'))
        return render_template('index.html', username=session['username'])

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = 'uploaded_video.mp4'
                file_path = f"uploads/{filename}"
                file.save(file_path)
                app.config['CURRENT_VIDEO'] = file_path
                global_state['restricted_area'] = None
                flash('Video uploaded successfully!')
                return redirect(url_for('auth_area_detection'))
        return render_template('upload.html')

    @app.route('/auth_area_detection')
    def auth_area_detection():
        if 'username' not in session:
            flash('You need to log in first!', 'danger')
            return redirect(url_for('index'))
        return render_template('auth_area_detection.html', username=session['username'])

    @app.route('/set_restricted_area', methods=['POST'])
    def set_restricted_area():
        x = int(request.form.get("x", 0))
        y = int(request.form.get("y", 0))
        w = int(request.form.get("w", 0))
        h = int(request.form.get("h", 0))
        global_state['restricted_area'] = (x, y, x + w, y + h)
        return '', 204

    @app.route('/update_gender', methods=['GET'])
    def update_gender():
        gender = request.args.get("gender", "both")
        global_state['selected_gender'] = gender
        return jsonify({"status": "updated", "selected_gender": gender})

    @app.route('/clear_detection_settings', methods=['POST'])
    def clear_detection_settings():
        global_state['restricted_area'] = None
        return "OK", 200

    @app.route('/toggle_counting', methods=['POST'])
    def toggle_counting():
        global_state['counting_enabled'] = not global_state['counting_enabled']
        return {"counting_enabled": global_state['counting_enabled']}

    @app.route('/video_feed')
    def video_feed():
        global_state['stop_video_flag'].clear()
        video_path = app.config.get('CURRENT_VIDEO', 'demo_browser/demo1.mp4')
        return Response(generate_frames(video_path, model, tracker, global_state), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/terminate_video_feed', methods=['POST'])
    def terminate_video_feed():
        global_state['stop_video_flag'].set()
        return redirect(url_for('auth_area_detection'))