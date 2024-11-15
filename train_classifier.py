import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import numpy as np

# Load the data
data_dict = pickle.load(open('imageData_and_labels/double_hand_data_word.pickle', 'rb'))

filtered_data = []
filtered_labels = []

for item, label in zip(data_dict['data'], data_dict['labels']):
    if np.array(item).shape != (42,):
        filtered_data.append(item)
        filtered_labels.append(label)

filtered_data = np.asarray(filtered_data)
filtered_labels = np.asarray(filtered_labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(filtered_data, filtered_labels, test_size=0.2, shuffle=True)

# Initialize the model
model = RandomForestClassifier()

# Train the model with tqdm progress bar
print("Training the model...")
model.fit(x_train, y_train)

# Make predictions
print("Making predictions...")
y_predict = None
y_predict = model.predict(x_test)

# Calculate the accuracy
score = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly!'.format(score * 100))
print("Classification report:")
print(classification_report(y_test,y_predict))

# Save the model
with open('saved_models/double_hand_model_word(scikit-upgraded).p', 'wb') as f:
    pickle.dump({'model': model}, f)
