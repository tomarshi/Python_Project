import pyttsx3
import speech_recognition as sr
import datetime
import webbrowser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  

commands = [
    "what time is it", "tell me the time", "current time", 
    "open google", "search google", "google", 
    "open youtube", "search youtube", "youtube", 
    "open facebook", "search facebook", "facebook",
    "open mit meerut", "search mit meerut", "mit meerut",
    "exit", "quit", "close",
]
categories = [
    "time", "time", "time",
    "web", "web", "web",
    "web", "web", "web",
    "web", "web", "web",
    "web", "web", "web",
    "exit", "exit", "exit"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(commands).toarray()

category_mapping = {category: index for index, category in enumerate(set(categories))}
y = [category_mapping[category] for category in categories]

model = LinearRegression()
model.fit(X, y)

def predict_category(command):
    X_test = vectorizer.transform([command]).toarray()
    prediction = model.predict(X_test)
    category_index = int(round(prediction[0]))
    category = [key for key, value in category_mapping.items() if value == category_index][0]
    return category

def talk(text):
    engine.say(text)
    engine.runAndWait()

def take_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
        except sr.UnknownValueError:
            talk("Sorry, I did not understand that.")
            return "None"
        except sr.RequestError as e:
            talk("Could not request results; check your network connection.")
            return "None"
    return command.lower()

def execute_command(command):
    category = predict_category(command)
    talk(f"The command category is {category}.")
    if category == "time":
        current_time = datetime.datetime.now().strftime('%I:%M %p')
        talk(f"The current time is {current_time}")
    elif category == "web":
        if "google" in command:
            webbrowser.open("https://www.google.com")
            talk("Opening Google")
        elif "youtube" in command:
            webbrowser.open("https://www.youtube.com")
            talk("Opening YouTube")
        elif "facebook" in command:
            webbrowser.open("https://www.facebook.com")
            talk("Opening Facebook")
        elif "mit meerut" in command:
            webbrowser.open("https://www.mitmeerut.ac.in")
            talk("Opening MIT Meerut")
    elif category == "exit":
        talk("Goodbye!")
        exit()
    else:
        talk("I am not sure how to help with that.")

def run_voice_assistant():
    talk("Hello! How can I help you today?")
    while True:
        command = take_command()
        if command and command != "None":
            execute_command(command)

if __name__ == "__main__":
    run_voice_assistant()
