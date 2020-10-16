import json
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.callbacks import BaseLogger

class TrainingMonitor(BaseLogger):

    def __init__(self, fig_path, json_path=None, start=0):
        super(TrainingMonitor, self).__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.start = start

    def on_train_begin(self, logs={}):
        self.H = {}
        if self.json_path and os.path.exists(self.json_path):
            self.H = json.loads(open(self.json_path).read())
            if start > 0:
                for key in self.H.keys():
                    self.H[key] = self.H[key][:self.start]

    def on_epoch_end(self, epoch, logs={}):
        for key, value in logs.items():
            data = self.H.get(key, [])
            data.append(float(value))
            self.H[key] = data

        if self.json_path:
            file = open(self.json_path, "w")
            file.write(json.dumps(self.H))
            file.close()

        plt.style.use("ggplot")
        plt.figure()
        N = np.arange(0, len(self.H["loss"]))
        plt.plot(N, self.H["loss"], label="training loss")
        plt.plot(N, self.H["val_loss"], label="validation loss")
        plt.plot(N, self.H["accuracy"], label="training accuracy")
        plt.plot(N, self.H["val_accuracy"], label="validation accuracy")
        plt.title(f"Training loss and accuracy [Epoch {len(self.H['loss'])}]")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(self.fig_path)
        plt.close()
