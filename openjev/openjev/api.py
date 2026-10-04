"""HTTP decision API.

    POST /v1/decide
    {
      "image": "<base64 PNG/JPEG/TIFF or .npy>"   # or
      "text":  "Customer says the invoice has a duplicate charge.",
      "questions": [{"type": "choice", "key": "queue", "question": "Which team?", "options": ["billing", "fraud"]},
                    {"type": "noul", "key": "urgent", "question": "Is this urgent?"}]
    }
    -> {"answers": {"queue": {"type": "choice", "value": "billing", "probs": {...}, "confidence": ...}, ...}}

Image requests go to a trained VisionJev (which answers the questions it was trained on; "questions" may
be omitted or select a subset by key). Text requests go to a text decider (LLMDecider or TextJevLite), which
accepts arbitrary questions.

    OPENJEV_VISION=model.pt OPENJEV_TEXT_LLM=Qwen/Qwen3-1.7B uvicorn openjev.api:app
"""
import base64
import io
import os

import numpy as np

from .schema import question_from_dict


def decode_image(b64):
    raw = base64.b64decode(b64)
    if raw[:6] == b"\x93NUMPY":
        return np.load(io.BytesIO(raw))
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(raw)))


def create_app(vision=None, text=None):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class Request(BaseModel):
        image: str | None = None
        text: str | None = None
        questions: list[dict] | None = None

    app = FastAPI(title="openjev")

    @app.get("/health")
    def health():
        return {"vision": vision is not None, "text": text is not None,
                "vision_questions": [q.key for q in vision.questions] if vision else []}

    @app.post("/v1/decide")
    def decide(req: Request):
        if (req.image is None) == (req.text is None):
            raise HTTPException(400, "provide exactly one of 'image' or 'text'")
        if req.image is not None:
            if vision is None:
                raise HTTPException(503, "no vision model loaded")
            answers = vision.predict(images=[decode_image(req.image)])[0]
            if req.questions:
                wanted = {q.get("key") for q in req.questions}
                unknown = wanted - set(answers)
                if unknown:
                    raise HTTPException(400, f"vision model was not trained on: {sorted(map(str, unknown))}")
                answers = {k: v for k, v in answers.items() if k in wanted}
            return {"answers": answers}
        if text is None:
            raise HTTPException(503, "no text model loaded")
        if not req.questions:
            raise HTTPException(400, "text requests need 'questions'")
        try:
            qs = [question_from_dict(q) for q in req.questions]
        except (KeyError, TypeError) as e:
            raise HTTPException(400, f"bad question: {e}")
        return {"answers": text.decide(req.text, qs)}

    return app


def _from_env():
    vision = text = None
    if os.environ.get("OPENJEV_VISION"):
        from .vision import VisionJev

        vision = VisionJev.load(os.environ["OPENJEV_VISION"])
    if os.environ.get("OPENJEV_TEXT_LLM"):
        from .text import LLMDecider

        text = LLMDecider(os.environ["OPENJEV_TEXT_LLM"])
    return create_app(vision, text)


def __getattr__(name):
    # lazily build `app` so importing this module does not load models
    if name == "app":
        return _from_env()
    raise AttributeError(name)
