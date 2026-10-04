"""Random-weight tiny models + offline tokenizers: exercise the decider plumbing without downloads."""
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

SPECIAL = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<|im_start|>", "<|im_end|>"]


def _tokenizer(extra=()):
    from transformers import PreTrainedTokenizerFast

    words = ["[PAD]", "[UNK]", "</s>"] + list(extra) + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    kw = {}
    if extra:
        kw["extra_special_tokens"] = {"image_token": "<|image_pad|>", "video_token": "<|video_pad|>",
                                      "vision_start_token": "<|vision_start|>", "vision_end_token": "<|vision_end|>"}
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]", eos_token="</s>", **kw), len(words)


def tiny_llm():
    from transformers import LlamaConfig, LlamaForCausalLM

    tok, v = _tokenizer()
    cfg = LlamaConfig(vocab_size=v, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=4, pad_token_id=0)
    return LlamaForCausalLM(cfg), tok


def tiny_vlm():
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    from transformers.models.qwen2_vl import Qwen2VLImageProcessorPil, Qwen2VLVideoProcessor

    tok, v = _tokenizer(SPECIAL)
    tmpl = ("{% for m in messages %}<|im_start|> {{ m['role'] }} "
            "{% if m['content'] is string %}{{ m['content'] }}{% else %}{% for c in m['content'] %}"
            "{% if c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ c['text'] }}{% endif %}"
            "{% endfor %}{% endif %} <|im_end|> {% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}")
    proc = Qwen2VLProcessor(image_processor=Qwen2VLImageProcessorPil(min_pixels=28 * 28 * 4, max_pixels=28 * 28 * 16),
                            tokenizer=tok, video_processor=Qwen2VLVideoProcessor(), chat_template=tmpl)
    ids = {t: tok.convert_tokens_to_ids(t) for t in SPECIAL}
    cfg = Qwen2VLConfig(
        text_config=dict(vocab_size=v, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, bos_token_id=None, eos_token_id=2,
                         rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1], "rope_type": "default"}),
        vision_config=dict(depth=1, embed_dim=32, hidden_size=32, num_heads=4, patch_size=14, spatial_merge_size=2,
                           in_chans=3, temporal_patch_size=2),
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"])
    return Qwen2VLForConditionalGeneration(cfg), proc
