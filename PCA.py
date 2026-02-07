import matplotlib
matplotlib.use('TkAgg') #This was just incase I needed to force
                        #matplotlib to open up a plot in OSwindows

import matplotlib.pyplot as plt
import numpy as np


from sklearn.datasets import fetch_openml #I forgot to pip install this onto my computer lol
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

mnist = fetch_openml('mnist_784', version = 1, as_frame = False)


X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

X = X[:42000] #These two caused it to take super long to load
y = y[:42000] #I messed with them to get faster runtimes

print(X.shape)
print(y.shape)

pca = PCA(n_components = 50, random_state = 42)
X = pca.fit_transform(X)

print(X.shape)

tsne = TSNE(
    n_components = 2,
    perplexity = 50,
    max_iter = 3000,
    init = 'pca',
    learning_rate = 'auto',
    random_state = 42
)

X_embedded = tsne.fit_transform(X)

plt.figure(figsize=( 8, 8))
scatter = plt.scatter(
    X_embedded[:, 0],
    X_embedded[:, 1],
    c = y,
    cmap = 'tab10',
    s = 1
)
plt.title("t-SNE on MNIST (42K points, perplexity=50)")
plt.colorbar(scatter)
plt.show()

perplexities = [30, 50, 100]
iterations = [1000, 3000]

for perp in perplexities:
    for iters in iterations:
        print(f"Running t-SNE: perplexity={perp}, iterations={iters}")
        tsne = TSNE(
            n_components = 2,
            perplexity = perp,
            max_iter = iters,
            init = 'pca',
            learning_rate = 'auto',
            random_state = 42
        )

        X_embedded = tsne.fit_transform(X)
        print("  Done.") #I included this
                        #to let me know it was still running

        plt.figure(figsize = (6, 6))
        plt.scatter(
            X_embedded[:, 0],
            X_embedded[:, 1],
            c = y,
            cmap = 'tab10',
            s = 1
        )
        plt.title(f"perplexity={perp}, iterations={iters}")
        plt.show() # I forgot to include this and scrambled for an hour as to why it wouldnt work

