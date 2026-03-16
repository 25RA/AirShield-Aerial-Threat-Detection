from flask import Flask, request, jsonify, render_template
import os
from utils.fusion_engine import fuse_audio_video
from utils.audio_infer import predict_audio
from utils.video_infer import detect_video

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect/audio", methods=["POST"])
def detect_audio():
    try:
        file = request.files["file"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        result = predict_audio(path)
        return jsonify(result)

    except Exception as e:
        print("AUDIO ERROR:", e)
        return jsonify({"error": "audio_failed"}), 500

@app.route("/detect/video", methods=["POST"])
def detect_video_api():
    try:
        file = request.files["file"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        result = detect_video(path)
        return jsonify(result)

    except Exception as e:
        print("VIDEO ERROR:", e)
        return jsonify({"error": "video_failed"}), 500

@app.route("/detect/fusion", methods=["POST"])
def detect_fusion():
    try:
        audio_file = request.files.get("audio")
        video_file = request.files.get("video")

        audio_result = {"class": "unknown", "confidence": 0.0}
        video_result = {"final_class": "unknown"}

        if audio_file:
            audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
            audio_file.save(audio_path)
            audio_result = predict_audio(audio_path)

            if video_file:
                video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
                video_file.save(video_path)
                video_result = detect_video(video_path)
    
            fusion_output = fuse_audio_video(audio_result, video_result)
            return jsonify(fusion_output)

    except Exception as e:
        print("FUSION ERROR:", e)
        return jsonify({"error": "fusion_failed"}), 500


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
