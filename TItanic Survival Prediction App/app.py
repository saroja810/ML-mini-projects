# Import libraries
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Titanic.csv")

# Remove unnecessary columns
data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Encode gender column
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

# Fill missing age values
data['Age'] = data['Age'].fillna(
    data.groupby('Sex')['Age'].transform('median')
)

# Features and target
X = data.drop(columns=["Survived"])
y = data["Survived"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale Age and Fare
scaler = StandardScaler()

X_train[['Age', 'Fare']] = scaler.fit_transform(
    X_train[['Age', 'Fare']]
)

X_test[['Age', 'Fare']] = scaler.transform(
    X_test[['Age', 'Fare']]
)

# Train models
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Calculate accuracy
lr_acc = accuracy_score(y_test, lr.predict(X_test))
dt_acc = accuracy_score(y_test, dt.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# Select best model
best_model = max(
    [(lr, lr_acc), (dt, dt_acc), (rf, rf_acc)],
    key=lambda x: x[1]
)[0]

best_acc = max([lr_acc, dt_acc, rf_acc])

# Streamlit App
st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.number_input("Age", step=1)
sibsp = st.number_input("Siblings / Spouses", step=1)
parch = st.number_input("Parents / Children", step=1)
fare = st.number_input("Fare")

if st.button("Predict"):

    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [0 if sex == 'male' else 1],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare]
    })

    input_data[['Age', 'Fare']] = scaler.transform(
        input_data[['Age', 'Fare']]
    )

    prediction = best_model.predict(input_data)

    st.write(f"Best Model: {best_model.__class__.__name__}")
    st.write(f"Accuracy: {best_acc:.2f}")

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")
