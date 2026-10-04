"""Zero-shot typed decisions on images with a CLIP/SigLIP-style image-text model (no training labels).

Each option becomes a text prompt; the image-text similarities are softmaxed into option probabilities.
Useful to bootstrap before labels exist, and to pre-label data for VisionJev. Probabilities are NOT
calibrated out of the box: fit one temperature per question on a small labelled set (fit_temperature).
"""
import numpy as np

from ..schema import Noul, Score, format_answer
from .. import calibration as cal
from .backbones import ALIASES, normalize_image


class ZeroShotVision:
    def __init__(self, model_id="siglip2", template="a microscopy image of {}.", device=None):
        import torch
        from transformers import AutoModel, AutoProcessor

        model_id = ALIASES.get(model_id, model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.template = template
        self.temperatures = {}

    def _prompts(self, q):
        if isinstance(q, Noul):
            return [q.no_text or f"not {q.question}", q.yes_text or q.question]
        return [self.template.format(o) for o in q.labels]

    def logits(self, images, q):
        import torch

        imgs = [(normalize_image(im) * 255).astype(np.uint8) for im in images]
        # SigLIP models are trained with max_length padding
        inputs = self.processor(text=self._prompts(q), images=imgs, padding="max_length",
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model(**inputs).logits_per_image.float().cpu().numpy()  # (N, n_options)

    @staticmethod
    def _softmax(z, T=1.0):
        z = z / T
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def fit_temperature(self, images, q, y):
        z = self.logits(images, q)
        self.temperatures[q.key] = cal.fit_temperature(lambda T: self._softmax(z, T), np.asarray(y))
        return self.temperatures[q.key]

    def predict(self, images, questions):
        P = {q.key: self._softmax(self.logits(images, q), self.temperatures.get(q.key, 1.0)) for q in questions}
        return [{q.key: format_answer(q, P[q.key][i]) for q in questions} for i in range(len(images))]
