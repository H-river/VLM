"""Training package for profile2setup v2."""

_HAS_TORCH_DATASET = True
try:
    from .dataset import (
        Profile2SetupDataset,
        filter_records,
        load_jsonl,
        profile2setup_collate_fn,
        validate_no_forbidden_fields,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _HAS_TORCH_DATASET = False
from .normalization import (
    CANONICAL_VARIABLE_ORDER,
    clamp_setup_to_ranges,
    denormalize_delta_vector,
    denormalize_setup_vector,
    get_variable_order,
    load_variables_config,
    make_zero_setup_vector,
    normalize_delta_vector,
    normalize_setup_vector,
)
from .preprocessing import (
    load_intensity,
    make_profile_channels,
    normalize_intensity,
    resize_intensity,
)
from .text import (
    SimpleTokenizer,
    build_vocab_from_jsonl,
    build_vocab_from_jsonl_files,
    load_vocab,
    save_vocab,
)

__all__ = [
    "CANONICAL_VARIABLE_ORDER",
    "SimpleTokenizer",
    "build_vocab_from_jsonl",
    "build_vocab_from_jsonl_files",
    "clamp_setup_to_ranges",
    "denormalize_delta_vector",
    "denormalize_setup_vector",
    "get_variable_order",
    "load_intensity",
    "load_variables_config",
    "load_vocab",
    "make_profile_channels",
    "make_zero_setup_vector",
    "normalize_delta_vector",
    "normalize_intensity",
    "normalize_setup_vector",
    "resize_intensity",
    "save_vocab",
]

if _HAS_TORCH_DATASET:
    __all__.extend(
        [
            "Profile2SetupDataset",
            "filter_records",
            "load_jsonl",
            "profile2setup_collate_fn",
            "validate_no_forbidden_fields",
        ]
    )
