import openai
import pyttsx3
import speech_recognition as sr

openai.api_key = ""

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
        text = recognizer.recognize(audio)
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

# Example usage
while True:
    user_input = speech_to_text()
    if user_input.strip().lower() == "exit":
        break
    response = send_message(user_input)
    print("Response:", response)
    text_to_speech(response)
