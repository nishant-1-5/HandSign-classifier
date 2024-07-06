import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# Loading training data
df = pd.read_csv("sign_mnist_train.csv")
df.dropna(inplace=True)
Y_train = np.array(df['label'].values)
X_train = np.array(df.drop(columns='label').values)

warnings.simplefilter(action='ignore', category=FutureWarning)
m,n = X_train.shape
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
fig.tight_layout(pad=0.1)

for i, ax in enumerate(axes.flat):
    random_index = np.random.randint(m)
    X_random_reshaped = X_train[random_index].reshape((28, 28)).T
    ax.imshow(X_random_reshaped, cmap='grey')
    ax.set_title(Y_train[random_index])
    ax.set_axis_off()

plt.show()


model = Sequential(
    [
        Input(shape=(X_train.shape[1],)),
        Dense(30, activation='relu'),
        Dense(20, activation='relu'),
        Dense(15, activation='relu'),
        Dense(25, activation='linear'),
    ]
)

print(model.summary())

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.00003),
    metrics=['accuracy']
              )

model.fit(X_train, Y_train, epochs=300, validation_split=0.2)

# test
test_df = pd.read_csv("sign_mnist_test.csv")
df.dropna(inplace=True)
y_test = np.array(df['label'].values)
X_test = np.array(df.drop(columns='label').values)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

logits = model.predict(X_test)
predictions = np.argmax(logits, axis=1)

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
fig.tight_layout(pad=0.1)

for i, ax in enumerate(axes.flat):
    random_index = np.random.randint(len(X_test))
    X_random_reshaped = X_test[random_index].reshape((28, 28)).T
    ax.imshow(X_random_reshaped, cmap='grey')
    ax.set_title(f"{y_test[random_index]} , {predictions[random_index]}")
    ax.set_axis_off()
plt.tight_layout()
plt.show()

