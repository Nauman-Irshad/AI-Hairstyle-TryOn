"""
HairCLIP-style interface (optional): text-driven hairstyle selection for future StyleGAN / CLIP integration.

This module does not add required dependencies. If `open_clip_torch` is installed, embeddings are available;
otherwise callers should use filename / id based hairstyle selection (default in this project).
"""

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class HairstyleDescription:
    style_id: str
    text_prompt: str


# Curated prompts aligned with dropdown IDs in the frontend (editable).
DEFAULT_STYLES: List[HairstyleDescription] = [
    HairstyleDescription("style_wavy_brown", "natural wavy brown hair"),
    HairstyleDescription("style_short_black", "short neat black hair"),
    HairstyleDescription("style_long_blonde", "long straight blonde hair"),
]


def list_builtin_styles() -> List[HairstyleDescription]:
    return list(DEFAULT_STYLES)


class HairCLIPAdapter:
    """
    Optional CLIP text encoder wrapper. Lazy import — no hard dependency on open_clip.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def is_available(self) -> bool:
        try:
            import open_clip  # noqa: F401
            return True
        except ImportError:
            return False

    def encode_text(self, prompt: str) -> Optional[Any]:
        """Returns normalized text embedding tensor on CPU, or None if CLIP unavailable."""
        if not self.is_available():
            return None
        import open_clip
        import torch

        if self._model is None:
            self._model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._model.eval()
        with torch.no_grad():
            t = self._tokenizer([prompt])
            emb = self._model.encode_text(t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu()
