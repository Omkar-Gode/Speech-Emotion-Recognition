from flask import Flask, render_template, request, redirect, url_for, flash
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import pickle

app = Flask(__name__)

UPLOAD_FOLDER = "static/Audio_files/"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

Scaler =  pd.read_csv("model_files/Scaling_Parameters_copy")

model = tf.keras.models.load_model('model_files/SER_better_model.h5')

lb_enc = pickle.load(open("model_files/Enc_labels.csv","rb"))

def zcr(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)


def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)

def extract_features(data, sr, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                                    ))
    return result

def get_features(path, duration=2.5, offset=0.6):
    data, sample_rate = librosa.load(path, duration=duration, offset=offset)
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    return result

def scale_data(array,mean,var):
    std=var **0.5
    return (array-mean)/std


@app.route("/")
def hello_world():
    return render_template('index_app.html')

@app.route("/result", methods=["GET","POST"])
def result():
    file = request.files["Sound"]
    filename = file.filename
    file.save(os.path.join(app.config["UPLOAD_FOLDER"],filename))

    feature = get_features("static/Audio_files/"+filename)
    fa = np.array(feature)
    fa = np.expand_dims(fa, axis=0)
    fz = np.zeros((fa.shape[0],2376-fa.shape[1]))
    X = np.append(fa,fz,axis=1)
    X_df = pd.DataFrame(X)

    mean1 = Scaler["Mean"]
    var1 = Scaler["Var"]

    X_scaled = scale_data(X_df.iloc[0,:],mean1,var1)
    X_scaled_df = pd.DataFrame(X_scaled)
    X_scaled_df = np.expand_dims(X_scaled_df, axis=0)

    y_pred = model.predict(X_scaled_df)
    percentage = round(y_pred.max()*100, 2)
    print(percentage)
    print(y_pred)
    print(y_pred.argmax())
    prediction = lb_enc.classes_[y_pred.argmax()]
    prediction = prediction.upper()
    print(lb_enc.classes_)
    print(prediction)

    return render_template('output_app.html',filename=filename, Prediction=prediction, Percentage=percentage)

@app.route("/play/<filename>")
def play_audio(filename):
    file_url = "Audio_files/"+filename
    return redirect(url_for('static',filename=file_url))

@app.route("/display/<Prediction>")
def display_emotion(Prediction):
    print("emotion function")
    if Prediction == "ANGRY":
        emotion_url="Emotions/angry.png"
    elif Prediction == "DISGUST":
        emotion_url="Emotions/disgust.png"
    elif Prediction == "FEAR":
        emotion_url="Emotions/fear.png"
    elif Prediction == "HAPPY":
        emotion_url="Emotions/happy.png"
    elif Prediction == "NEUTRAL":
        emotion_url="Emotions/neutral.png"
    else:
        emotion_url="Emotions/sad.png"
    print(emotion_url)

    return redirect(url_for('static', filename=emotion_url))

if __name__=="__main__":
    app.run(debug=True)