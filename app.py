import os
import uuid
import logging
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import whisper
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
client = OpenAI(api_key=api_key)

# Load Whisper model
model = whisper.load_model("base")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg"}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(file_path):
    """Transcribe an audio file using Whisper."""
    try:
        result = model.transcribe(file_path)
        return result.get("text", "")
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return "Error during transcription."

def summarize_text(text):
    """Summarize the transcribed text using OpenAI GPT API."""
    try:
        prompt = f"Please provide a concise summary of the following meeting transcript:\n\n{text}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes meeting transcripts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=40000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return "Error during summarization."

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Get the uploaded file
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            try:
                # Create a secure filename with a UUID to prevent conflicts
                filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)

                # Transcribe and summarize the audio
                transcript = transcribe_audio(file_path)
                summary = summarize_text(transcript)

                # Delete the file after processing
                os.remove(file_path)

                return render_template("index.html", transcript=transcript, summary=summary)
            except Exception as e:
                logging.error(f"Error processing file: {e}")
                flash("An error occurred while processing your file. Please try again.")
                return redirect(request.url)
        else:
            flash("Invalid file type. Please upload an audio file (wav, mp3, m4a, ogg).")
            return redirect(request.url)

    return render_template("index.html", transcript=None, summary=None)

if __name__ == "__main__":
    app.run(debug=True)
