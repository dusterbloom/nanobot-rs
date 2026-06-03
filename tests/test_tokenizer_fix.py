"""
Test that the MLX model's config.json includes transformers_version to prevent
false-positive Mistral regex warnings for Qwen3 models.

The _patch_mistral_regex check in transformers triggers on any model with
vocab > 100K and a pretokenizer. Qwen3 (248K vocab) hits this but isn't
actually Mistral. Adding transformers_version > 4.57.3 makes the check
return early, avoiding both the warning and the kwarg-duplication crash.
"""
import json
from pathlib import Path
from packaging import version


MODEL_DIR = Path.home() / ".cache/lm-studio/models/Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-4bit"


def test_config_has_transformers_version():
    """Model config.json must have transformers_version to bypass false Mistral regex check."""
    config_path = MODEL_DIR / "config.json"
    assert config_path.exists(), f"config.json not found at {config_path}"

    with open(config_path) as f:
        config = json.load(f)

    tv = config.get("transformers_version")
    assert tv is not None, (
        "config.json must have transformers_version to prevent "
        "false-positive Mistral regex warnings for Qwen3 models"
    )
    assert version.parse(tv) > version.parse("4.57.3"), (
        f"transformers_version must be > 4.57.3 to bypass the Mistral regex check, got {tv}"
    )


def test_model_type_is_not_mistral():
    """Confirm model_type is qwen3, not a Mistral variant."""
    config_path = MODEL_DIR / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "")
    mistral_types = {"mistral", "mistral3", "voxtral", "ministral", "pixtral"}
    assert model_type not in mistral_types, (
        f"model_type={model_type} is Mistral - regex fix would be needed"
    )


def test_tokenizer_loads_without_mistral_warning():
    """Tokenizer must load without triggering the Mistral regex warning."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        mistral_warns = [x for x in w if "mistral" in str(x.message).lower() and "regex" in str(x.message).lower()]
        assert len(mistral_warns) == 0, (
            f"Mistral regex warning still present: {mistral_warns[0].message}"
        )
        assert tok.vocab_size > 0, "Tokenizer should load with valid vocab"


if __name__ == "__main__":
    test_config_has_transformers_version()
    print("PASS: test_config_has_transformers_version")
    test_model_type_is_not_mistral()
    print("PASS: test_model_type_is_not_mistral")
    test_tokenizer_loads_without_mistral_warning()
    print("PASS: test_tokenizer_loads_without_mistral_warning")
