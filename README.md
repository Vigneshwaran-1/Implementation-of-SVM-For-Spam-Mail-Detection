# Implementation-of-SVM-For-Spam-Mail-Detection

## Developed by : Vigneshwaran P
## Register no: 212224040358

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
~~~
1. Start and import the required libraries.
2. Create the email dataset with spam and non-spam labels.
3. Convert the email text into numerical features.
4. Train the SVM model and predict the test emails.
5. Display the accuracy and visualize the result using a graph.
~~~
## Program:
```

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

emails = [
    "Win a free lottery prize now",
    "Congratulations you won money",
    "Claim your free gift",
    "Meeting is scheduled for tomorrow",
    "Please submit the assignment",
    "Let's discuss the project",
    "You have won a free cash prize",
    "Class starts at 10 AM",
    "Buy now and get a free offer",
    "See you tomorrow in college"
]

labels = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Actual Labels:", y_test)
print("Predicted Labels:", y_pred)
print("Accuracy:", accuracy)

plt.bar(["Actual Spam", "Predicted Spam"],
        [sum(y_test), sum(y_pred)])

plt.title("SVM Spam Mail Detection")
plt.ylabel("Number of Spam Emails")
plt.show()
```

## Output:
<img width="473" height="320" alt="image" src="https://github.com/user-attachments/assets/d6e433f7-b1cc-4d12-a2f3-32577b718414" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
