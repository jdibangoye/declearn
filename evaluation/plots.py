# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(xs, ys, cis, path: str):
    plt.figure()
    xs = np.array(xs)
    ys = np.array(ys)
    cis = np.array(cis)
    plt.plot(xs, ys)
    plt.fill_between(xs, ys - cis, ys + cis, alpha=0.3)
    plt.xlabel("steps")
    plt.ylabel("return")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()