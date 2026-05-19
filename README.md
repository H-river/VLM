# VLM Optical Setup Workflows

This repo has two active project areas:

- `optical_sim/`: simulator and data-generation code for optical beam profiles.
- `profile2setup/`: the main profile-to-setup project.

The current main `profile2setup` approach is the multimodal LLM/API + SFT
workflow. Start with:

```bash
python -m profile2setup.scripts.check_v2_integrity_cli
python -m profile2setup.scripts.render_llm_api_images_cli --help
python -m profile2setup.scripts.build_llm_api_sft_dataset_cli --help
python -m profile2setup.scripts.audit_llm_sft_data_cli --help
python -m profile2setup.scripts.create_llm_api_sft_job_cli --help
python -m profile2setup.scripts.check_llm_api_sft_job_cli --help
python -m profile2setup.scripts.run_llm_api_inference_cli --help
python -m profile2setup.scripts.evaluate_llm_api_predictions_cli --help
```

Core current files are under `profile2setup/llm_api/`,
`profile2setup/data_prep/build_llm_api_sft_dataset.py`,
`profile2setup/evaluation/llm_api_eval.py`,
`profile2setup/evaluation/llm_api_visualization.py`, and
`profile2setup/experiments/llm_api_setup_understanding.py`.

The local PyTorch `profile2setup` model remains available as a baseline under
`profile2setup/models/`, `profile2setup/training/`, and
`profile2setup/inference/`. Older text-to-discrete-bin work, local reasoning VLM
work, and physics-understanding side experiments live under `legacy/`.

