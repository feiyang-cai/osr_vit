import numpy as np
#from tsnecuda import TSNE
import seaborn as sns
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = np.load("./data.npz")
in_embs = data['embs'][:4000]
out_embs = data['embs'][4000:8000]

in_targets = data['targets'][:4000]
out_targets = data['targets'][4000:8000]

#center_targets = np.array([i for i in range(6)]).reshape(6, )
#targets = np.concatenate([targets, center_targets], axis=0)[:5000]

embs = np.concatenate([out_embs, in_embs], axis=0)
targets = np.concatenate([out_targets, in_targets], axis=0)


#X_2d = TSNE().fit_transform(embs)
#X_2d = TSNE(perplexity=64.0, learning_rate=270).fit_transform(embs)
X_2d = TSNE(n_components=2, learning_rate='auto', early_exaggeration = 12, init='pca', perplexity=20, n_iter=10000).fit_transform(embs)

X_2d = X_2d[3200:]
targets[:4000] = 6
targets = targets[3200:]
print(X_2d.shape, targets.shape)

df = pd.DataFrame()
df["y"] = targets
df["comp-1"] = X_2d[:,0]
df["comp-2"] = X_2d[:,1]


#sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                palette=sns.color_palette("hls", 7),
#                data=df).set(title="Iris data T-SNE projection") 
sc = plt.scatter(x=X_2d[:, 0], y=X_2d[:, 1], c=targets)
plt.colorbar(sc)

import tikzplotlib

tikzplotlib.Flavors.latex.preamble()
#tikzplotlib.clean_figure()
tikzplotlib.save("test.tex")
plt.savefig('demo.png')