"""
Monkey-patch for mlx-lm to fix KV cache reuse on hybrid models (Qwen3.5).

Problem: mlx-lm's server uses `can_trim_prompt_cache()` which calls
`all(c.is_trimmable() for c in cache)`. ArraysCache (used by GDN/recurrent
layers) inherits `is_trimmable() -> False` from _BaseCache, so the ENTIRE
cache is rejected — even though KVCache layers CAN be trimmed.

This causes full prompt re-processing on every turn for Qwen3.5 and any
hybrid model mixing ArraysCache + KVCache.

Fix: patch can_trim_prompt_cache to use `any()` and trim_prompt_cache to
handle per-layer trimming. Non-trimmable layers (ArraysCache) are reset
so attention layers still benefit from cached KV state.

Usage:
    import scripts.mlx_hybrid_cache_fix  # before starting mlx-lm server

See: https://github.com/ml-explore/mlx-lm/issues/903
"""

import logging

logger = logging.getLogger(__name__)


def apply():
    """Apply the hybrid cache fix to mlx-lm's cache module."""
    try:
        from mlx_lm.models import cache as cache_mod
        from mlx_lm.models.cache import ArraysCache, _BaseCache
    except ImportError:
        logger.warning("mlx-lm not installed, skipping hybrid cache fix")
        return False

    # --- 1. Give ArraysCache a safe trim() that resets recurrent state ---
    if not hasattr(ArraysCache, "_orig_is_trimmable"):

        def _arrays_trim(self, n):
            # GDN recurrent state can't be partially trimmed.
            # Reset to empty so the model rebuilds it for the new tokens.
            # Attention layers (KVCache) still provide full prefix context.
            self.cache = [None] * len(self.cache)
            return n  # Report trimmed so offset accounting stays consistent

        ArraysCache._orig_is_trimmable = ArraysCache.is_trimmable
        ArraysCache.is_trimmable = lambda self: True
        ArraysCache.trim = _arrays_trim
        logger.info("Patched ArraysCache.is_trimmable + trim")

    # --- 2. Patch can_trim_prompt_cache: any() instead of all() ---
    _orig_can_trim = cache_mod.can_trim_prompt_cache

    def _can_trim_hybrid(cache):
        # Allow trimming if ANY layer supports it (KVCache layers).
        return any(c.is_trimmable() for c in cache)

    cache_mod.can_trim_prompt_cache = _can_trim_hybrid

    # Also patch the module-level import in the server
    try:
        import mlx_lm.server as server_mod

        server_mod.can_trim_prompt_cache = _can_trim_hybrid
    except (ImportError, AttributeError):
        pass

    # --- 3. Patch trim_prompt_cache for per-layer handling ---
    _orig_trim = cache_mod.trim_prompt_cache

    def _trim_hybrid(cache, num_tokens):
        if not _can_trim_hybrid(cache) or len(cache) == 0:
            return 0
        trimmed = 0
        for c in cache:
            result = c.trim(num_tokens)
            trimmed = max(trimmed, result)
        return trimmed

    cache_mod.trim_prompt_cache = _trim_hybrid

    try:
        server_mod.trim_prompt_cache = _trim_hybrid
    except (NameError, AttributeError):
        pass

    logger.info(
        "mlx-lm hybrid cache fix applied — "
        "Qwen3.5/hybrid models will reuse KVCache across turns"
    )
    return True


# Auto-apply on import
_applied = apply()
