

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

data_danger = [
    ("Help! I'm being followed by someone.", 1),
    ("I'm lost in the forest and it's getting dark.", 1),
    ("I saw someone lurking around my house at night.", 1),
    ("I'm feeling uneasy about my surroundings.", 1),
    ("I'm being followed by a suspicious person.", 1),
    ("I need help, I'm in an unsafe situation.", 1),
    ("Someone is following me.", 1),
    ("My life is in danger!", 1),
    ("I feel threatened by someone.", 1),
    ("I'm scared and need assistance.", 1),
    ("I'm in a dangerous situation and need help.", 1),
    ("I'm being chased by a stranger.", 1),
    ("I'm experiencing a panic attack.", 1),
    ("I'm being harassed by someone.", 1),
    ("I'm being stalked.", 1),
    ("I'm being threatened.", 1),
    ("I'm in trouble, please help me.", 1),
    ("I'm feeling unsafe.", 1),
    ("i am feeling depressed.", 1),
    ("I'm in an emergency.", 1),
    ("I'm in distress.", 1)
]

data_safe = [
    ("I fell down the stairs but I'm okay.", 0),
    ("I think I left my keys in the car.", 0),
    ("My cat is stuck in the tree.", 0),
    ("I heard a noise outside my house.", 0),
    ("I'm cooking dinner at home.", 0),
    ("I'm watching TV with my family.", 0),
    ("I'm taking a relaxing bath.", 0),
    ("I'm enjoying a peaceful walk.", 0),
    ("I'm having a good day.", 0),
    ("I'm reading a book.", 0),
    ("I'm having a conversation with a friend.", 0),
    ("I'm listening to music.", 0),
    ("I'm spending time outdoors.", 0),
    ("I'm doing yoga.", 0),
    ("I'm playing a game.", 0),
    ("I'm working on a project.", 0),
    ("I'm enjoying nature.", 0),
    ("I'm attending a social event.", 0),
    ("I'm sleeping soundly.", 0),
    ("I'm feeling relaxed.", 0)
]

data = data_danger + data_safe
np.random.shuffle(data)

X = [text for text, label in data]
y = [label for text, label in data]

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, stratify=y, random_state=42)

logistic_regression = LogisticRegression()
svm = SVC(probability=True)
random_forest = RandomForestClassifier()

voting_classifier = VotingClassifier(estimators=[('lr', logistic_regression), ('svm', svm), ('rf', random_forest)], voting='soft')

voting_classifier.fit(X_train, y_train)

example_text = str(input("Enter: "))
example_vectorized = vectorizer.transform([example_text])
prediction = voting_classifier.predict(example_vectorized)[0]
print("Example Prediction (1 - Danger, 0 - Not in danger):", prediction)


y_pred = voting_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy*100)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))






