import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open('isl_data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

processed_data = []
for sample in data_dict['data']:
    flat = np.array(sample).flatten()
    processed_data.append(flat)

data = np.array(processed_data)
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"{score * 100:.2f}% of the samples were classified correctly!")

with open('isl_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
