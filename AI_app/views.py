from django.shortcuts import render

# Create your views here.
def index_page(request):
    return render(request, "index.html")


def about_page(request):
    return render(request, "about.html")


def contact_page(request):
    return render(request, "contact.html")



def senti_page(request):
    return render(request, "sentiment.html")

import os
import cv2
import librosa
import numpy as np
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
from keras.models import load_model
from AI_project.settings import *
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image

# Load models
audio_model = load_model('model_checkpoint.h5')
image_video_model = load_model('model_optimal_image.h5')

# Label Encoders
audio_classes = ['Anger', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
audio_label_encoder = LabelEncoder()
audio_label_encoder.classes_ = np.array(audio_classes)

image_video_classes = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

def extract_audio_features(file_path):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    result = np.hstack((mfccs, chroma, mel))
    return result.reshape(1, -1)
from textblob import TextBlob
def process_image(file_path):
    targetx, targety = 96, 96
    img = image.load_img(file_path, target_size=(targetx, targety))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = image_video_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    return image_video_classes[predicted_class_index], float(np.max(predictions))

import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    targetx, targety = 96, 96
    frame_predictions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (targetx, targety))
        img_array = image.img_to_array(resized_frame)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        predictions = image_video_model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        frame_predictions.append(predicted_class_index)
        
        # Print emotion for each frame
        emotion = image_video_classes[predicted_class_index]
        print(f"Detected Emotion: {emotion}")

    cap.release()
    if frame_predictions:
        majority_prediction = max(set(frame_predictions), key=frame_predictions.count)
        return image_video_classes[majority_prediction], 1.0 
    return "Unknown", 0.0


from transformers import pipeline
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

@csrf_exempt
def analyze_sentiment(request):
    if request.method == "POST":
        file = request.FILES.get("file")
        text_input = request.POST.get("text")
        print(text_input)
        if text_input:
            result = emotion_classifier(text_input)
            emotion = result[0]["label"]
            confidence = result[0]["score"]


            return JsonResponse({"sentiment": emotion, "confidence": abs(confidence)})

        if file:
            file_name = default_storage.save(file.name, file)
            file_path = default_storage.path(file_name)
            
            if file.content_type.startswith("audio/"):
                features = extract_audio_features(file_path)
                predictions = audio_model.predict(features)
                predicted_class_index = np.argmax(predictions)
                sentiment = audio_label_encoder.classes_[predicted_class_index]
                confidence = float(np.max(predictions))
            
            elif file.content_type.startswith("image/"):
                sentiment, confidence = process_images(file_path)
            
            elif file.content_type.startswith("video/"):
                sentiment, confidence = process_videos(file_path)
            
            else:
                return JsonResponse({"error": "Unsupported file type"}, status=400)
            
            os.remove(file_path)
            return JsonResponse({"sentiment": sentiment, "confidence": confidence})

    return JsonResponse({"error": "Invalid request"}, status=400)
