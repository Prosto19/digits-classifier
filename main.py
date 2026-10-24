from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
print(len(digits.data))
print(type(digits))
print(type(digits.images))
print(digits.images.shape)
print(digits.images[0])
plt.imshow(digits.images[0],cmap = 'gray')
plt.show()

A = [0, 16, 16, 0]  # Цифра 0
B = [16, 0, 0, 16]  # Цифра 1
Y = [14, 2, 1, 15] # Неизвестная цифра

def distance(first, second):
    result = 0

    for i in range(len(first)):
        result += abs(first[i] - second[i])

    return result


images = [A, B]

best_distance = distance(Y, images[0])
best_index = 0

for i in range(1, len(images)):
    current_distance = distance(Y, images[i])

    if current_distance < best_distance:
        best_distance = current_distance
        best_index = i

print(best_index)
