from flask import Flask, jsonify, request
import werkzeug
import os
from flask_cors import CORS
import librosa
from pydub import AudioSegment
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import extractive_summarization

app = Flask(__name__)

CORS(app, origins='*')

@app.route("/")
def helloWorld():
    return "Audio Summarization"

@app.route('/textSum', methods=['POST'])
def textSummarization():
    text = request.form.get('inputText')
    print(text)
    
    extractive_summary = extractive_summarization.extractive_summarizer(text)
    print('Extractive Summary: ',extractive_summary)
    
    return extractive_summary
    



ALLOWED_EXTENSIONS = {'wav', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/audioSum", methods=["POST"])
def audioSummarization():
    if request.method == "POST":
        # Support for multiple files
        audioFiles = request.files.getlist('audio')
        
        # Filter out unsupported file types
        valid_files = [audio for audio in audioFiles if allowed_file(audio.filename)]
        
        if not valid_files:
            return jsonify({'error': 'No valid audio files provided. Supported formats: .wav, .flac'})

        # Load the speech-to-text model and processor
        processor = AutoProcessor.from_pretrained("AqeelShafy7/Whisper-Sinhala_Audio_to_Text")
        model = AutoModelForSpeechSeq2Seq.from_pretrained("AqeelShafy7/Whisper-Sinhala_Audio_to_Text")
        
        full_transcription = []
        
        for audioFile in valid_files:
            filename = werkzeug.utils.secure_filename(audioFile.filename)
            save_path = "./uploadedaudio/" + filename
            audioFile.save(save_path)
            # Process each audio file with librosa and the model
            audio_data, sampling_rate = librosa.load(save_path, sr=16000, mono=True)
            input_features = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt").input_features
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            # Ensure transcription ends with a full stop
            if not transcription[0].endswith('.'):
                transcription[0] += '.'
                
            full_transcription.append(transcription[0])
        # Combine the transcriptions into a single paragraph
        full_text = ' '.join(full_transcription)
        print(full_text)
        return jsonify({'message': full_text})
    else:
        return jsonify({'about': "Error occurred"})

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except:
        print("Unexpected Error")