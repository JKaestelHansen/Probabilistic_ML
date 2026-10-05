"""A tiny synthetic text-decision benchmark (support-ticket routing / urgency / sentiment), used in tests and as
a format example for real benchmarks. Real comparisons should use held-out human-labelled data.

Format: list of (state, question, label). The same examples can be sent to LLMDecider, TextJevLite or a
hosted decision API to compare accuracy and calibration.
"""
import numpy as np

from ..schema import Choice, Noul, Score

QUEUE = Choice("queue", "Which team should handle this ticket?", ["billing", "technical", "shipping", "fraud"])
URGENT = Noul("urgent", "Does this need a response within the hour?")
SENTIMENT = Score("sentiment", "How satisfied is the customer?", ["very unhappy", "unhappy", "neutral", "happy"])

_TOPICS = {
    "billing": ["I was charged twice for order {n}", "my invoice {n} shows the wrong amount",
                "please refund the duplicate payment on {n}", "the subscription fee went up without notice"],
    "technical": ["the app crashes when I open settings", "I cannot log in, it says error {n}",
                  "the page keeps loading forever", "sync stopped working after the update"],
    "shipping": ["my package {n} has not arrived", "the tracking for {n} has not moved in a week",
                 "the parcel was delivered to the wrong address", "the box arrived damaged"],
    "fraud": ["there are purchases on my account I did not make", "someone changed my password and email",
              "I got a phishing email pretending to be you", "unknown card payments appeared, order {n}"],
}
_MOOD = {0: "This is unacceptable and I am furious.", 1: "I am disappointed.", 2: "", 3: "Thanks, otherwise great service!"}
_URGENT = ["I need this fixed right now, it is blocking my business.", "Please respond immediately."]


def make_ticket_benchmark(n=400, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    topics = list(_TOPICS)
    for _ in range(n):
        topic = topics[rng.integers(4)]
        mood = int(rng.integers(4))
        urgent = bool(rng.random() < 0.3)
        text = rng.choice(_TOPICS[topic]).format(n=rng.integers(1000, 9999)).capitalize() + ". " + _MOOD[mood]
        if urgent:
            text += " " + rng.choice(_URGENT)
        out += [(text, QUEUE, topic), (text, URGENT, urgent), (text, SENTIMENT, SENTIMENT.levels[mood])]
    return out
