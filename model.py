from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = KNeighborsClassifier(
    n_neighbors=1,
    metric="manhattan"
)
model.fit(X_train, y_train)
i = 0

fig, ax = plt.subplots()

def show_digit():
    ax.clear()

    prediction = model.predict(X_test[i:i+1])

    ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
    ax.set_title(
        f"Картинка {i+1}/{len(X_test)}\n"
        f"Предсказание: {prediction[0]} | Ответ: {y_test[i]}"
    )
    ax.axis("off")
    fig.canvas.draw_idle()

def on_key(event):
    global i

    if event.key == "right":
        i = (i + 1) % len(X_test)
    elif event.key == "left":
        i = (i - 1) % len(X_test)
    else:
        return

    show_digit()

fig.canvas.mpl_connect("key_press_event", on_key)

show_digit()
plt.show()
