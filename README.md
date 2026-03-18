# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load and Prepare Data:

Import the dataset containing emails labeled as spam or ham, and split it into training and testing data.

2.Convert Text to Features:

Transform email text into numerical form using techniques like TF-IDF vectorization.

3.Train the SVM Model:

Apply the Support Vector Machine (SVM) algorithm on the training data to learn patterns.

4.Test and Evaluate Model:

Predict results on test data and calculate accuracy to check performance.

## Program:
```
DEVELOPED BY: PAARKAVI A
REG NO: 212225040292
# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------------
# Step 1: Create dataset
# ------------------------------
data = {
    'v1': ['ham','ham','spam','ham','ham'],
    'v2': [
        'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...',
        'Ok lar... Joking wif u oni...',
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        'U dun say so early hor... U c already then say...',
        'Nah I don\'t think he goes to usf, he lives around here though'
    ]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Encode labels (ham=0, spam=1)
# ------------------------------
df['label'] = df['v1'].map({'ham':0, 'spam':1})

# ------------------------------
# Step 3: Feature extraction (TF-IDF)
# ------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['v2'])
y = df['label']

# ------------------------------
# Step 4: Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Step 5: Train SVM classifier
# ------------------------------
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# ------------------------------
# Step 6: Make predictions
# ------------------------------
y_pred = svm_model.predict(X_test)

# ------------------------------
# Step 7: Evaluate the model
# ------------------------------
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# Step 8: Predict new message
# ------------------------------
new_message = ["Congratulations! You have won a free ticket to Bahamas. Call now!"]
new_message_vect = vectorizer.transform(new_message)
prediction = svm_model.predict(new_message_vect)
print(f"Prediction: {'Spam' if prediction[0]==1 else 'Ham'}")

```

## Output:

<img width="837" height="488" alt="image" src="https://github.com/user-attachments/assets/1764409b-996d-4101-bd62-9bf8f5e084b9" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
