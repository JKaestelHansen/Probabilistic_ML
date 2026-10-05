# %% [markdown]
# # openjev quickstart: image QC + morphology decisions
# Run cell-by-cell in VS Code / Jupyter. Swap `make_blob_dataset` for your own images + labels.

# %%
import json
import sys

sys.path.insert(0, "..")
from openjev import Choice, Noul, Score
from openjev.data import make_blob_dataset
from openjev.vision import VisionJev

# %%
images, labels = make_blob_dataset(800, seed=0)
questions = [
    Choice("morphology", "What is the dominant shape of the objects?", ["round", "elongated", "irregular"]),
    Score("focus", "How well focused is the image?", ["very blurry", "blurry", "soft", "sharp"]),
    Noul("usable", "Is this image usable for analysis?"),
]

# %% backbone="dinov2" / "siglip2" / "hf:<id>" once Hugging Face downloads are available
jev = VisionJev(questions, backbone="handcrafted", n_members=5)
X = jev.embed(images)  # cache: the backbone is frozen, so this is the only expensive step

# %%
train, test = slice(0, 600), slice(600, None)
sub = lambda s: {k: v[s] for k, v in labels.items()}
jev.fit(X[train], sub(train))
print(json.dumps(jev.evaluate(X[test], sub(test)), indent=2))

# %% typed answers for one image
print(json.dumps(jev.predict(features=X[600:601])[0], indent=2))

# %% reliability diagram for the usable gate
import matplotlib.pyplot as plt
import numpy as np
from openjev import calibration as cal

probs, _ = jev.predict_proba(X[test])["usable"]
y = np.array(sub(test)["usable"], dtype=int)
conf, acc, count = cal.reliability_bins(probs, y)
m = count > 0
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.plot(conf[m], acc[m], "o-")
plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.title("usable: reliability")
plt.show()
