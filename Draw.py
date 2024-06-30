import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from LeafNetwork import LeafNetwork

# Charger le modèle
nn = LeafNetwork(784)
nn.load("mnist_model.json")

# Fonction pour dessiner et prédire
class DrawAndPredict:
    def __init__(self, model):
        self.model = model
        self.drawing = False
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = self.ax.figure.canvas
        self.image = np.zeros((28, 28))
        self.ax.imshow(self.image, cmap='gray')
        self.cid = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Bouton pour prédire
        self.ax_button_predict = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.btn_predict = Button(self.ax_button_predict, 'Predict')
        self.btn_predict.on_clicked(self.predict)

        # Bouton pour effacer
        self.ax_button_clear = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.btn_clear = Button(self.ax_button_clear, 'Clear')
        self.btn_clear.on_clicked(self.clear)

        plt.show()

    def on_press(self, event):
        self.drawing = True
        self.update_image(event)

    def on_release(self, event):
        self.drawing = False

    def on_motion(self, event):
        if self.drawing:
            self.update_image(event)

    def update_image(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < 28 and 0 <= y < 28:
            self.image[y-1:y+2, x-1:x+2] = 1.0
            self.ax.imshow(self.image, cmap='gray')
            self.canvas.draw()

    def predict(self, event):
        img = self.image.reshape(1, 784)
        img = img / 255.0  # Normalisation
        prediction = self.model.predict(img)
        predicted_label = np.argmax(prediction)
        print(f'Predicted Label: {predicted_label}')

    def clear(self, event):
        self.image = np.zeros((28, 28))
        self.ax.imshow(self.image, cmap='gray')
        self.canvas.draw()

# Initialiser l'outil de dessin et de prédiction
draw_and_predict = DrawAndPredict(nn)
