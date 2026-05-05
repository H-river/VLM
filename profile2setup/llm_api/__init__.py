"""LLM API schema and validation helpers for profile2setup."""

from .client import (
    SUPPORTED_PROVIDERS,
    build_openai_request_payload,
    call_multimodal_model,
)
from .sft_jobs import (
    create_sft_job,
    load_job_metadata,
    retrieve_sft_job,
    save_job_metadata,
    update_job_metadata,
    upload_training_file,
    upload_validation_file,
    validate_sft_jsonl,
)
from .schema import (
    CANONICAL_VARIABLE_ORDER,
    EXPECTED_TOP_LEVEL_FIELDS,
    LEGACY_VARIABLES,
    VALID_TASK_TYPES,
    default_output_schema,
)
from .sft_records import (
    build_assistant_label,
    build_sft_record,
    build_user_message,
    image_file_to_data_url,
    write_jsonl,
)
from .inference import (
    load_jsonl_records,
    render_or_reuse_images,
    run_llm_api_inference,
)
from .image_rendering import (
    load_intensity_npy,
    normalize_intensity,
    render_composite_png,
    render_difference_png,
    render_profile_png,
)
from .prompts import (
    build_messages,
    build_system_prompt,
    build_user_content,
)
from .validator import (
    parse_json_text,
    parse_llm_json_text,
    validate_all_variable_dicts,
    validate_llm_output,
    validate_predicted_delta,
    validate_predicted_setup,
    validate_variable_dict,
)

__all__ = [
    "CANONICAL_VARIABLE_ORDER",
    "EXPECTED_TOP_LEVEL_FIELDS",
    "LEGACY_VARIABLES",
    "SUPPORTED_PROVIDERS",
    "VALID_TASK_TYPES",
    "build_assistant_label",
    "build_messages",
    "build_openai_request_payload",
    "build_sft_record",
    "build_system_prompt",
    "build_user_content",
    "build_user_message",
    "call_multimodal_model",
    "create_sft_job",
    "default_output_schema",
    "image_file_to_data_url",
    "load_intensity_npy",
    "load_job_metadata",
    "load_jsonl_records",
    "normalize_intensity",
    "parse_json_text",
    "parse_llm_json_text",
    "render_composite_png",
    "render_difference_png",
    "render_or_reuse_images",
    "render_profile_png",
    "retrieve_sft_job",
    "run_llm_api_inference",
    "save_job_metadata",
    "update_job_metadata",
    "upload_training_file",
    "upload_validation_file",
    "validate_all_variable_dicts",
    "validate_llm_output",
    "validate_predicted_delta",
    "validate_predicted_setup",
    "validate_sft_jsonl",
    "validate_variable_dict",
    "write_jsonl",
]
