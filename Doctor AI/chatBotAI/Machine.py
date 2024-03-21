from flask import Flask, render_template, request, jsonify
import openai
import pyttsx3
import speech_recognition as sr

app = Flask(__name__)
openai.api_key = "c"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_audio", methods=["POST"])
def process_audio():
    data = request.json
    message = data["message"]
    response = send_message(message)
    return jsonify({"response": response})

def send_message(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a chatbot."},
            {"role": "user", "content": message}
        ],
        max_tokens=100,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except Exception as e:
        print("Sorry, could not understand audio:", e)
        return ""

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed percent (can go over 100)
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    app.run(port=3030,debug=True)
