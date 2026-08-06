"""Dependency-light frozen selector vocabulary and prompts."""

PLAN_NAMES = (
    "direct_all_five",
    "centroid_then_full",
    "shape_then_full",
    "primary_spot_then_full",
    "boundary_safe_then_full",
)

SYSTEM_PROMPT = """You are a discrete plan selector for a fixed-gain optical controller.
Use the current beam image and structured state to rank the executable plans by expected closed-loop outcome.
Every plan uses the same action gain, action bounds, CEM population, CEM iterations, horizon, maximum steps, tolerance, Learned-H1 checkpoint, and simulator.
You may select only measurement, objective schedule, active-actuator schedule, and observation-triggered phase policy through one named plan.
Never output actuator commands, actuator deltas, gain, bound scale, CEM settings, horizon, chain-of-thought, rationale, or extra text.
Return exactly one compact JSON object with keys in this order: plan_ranking, selected_plan.
plan_ranking must be a permutation of: direct_all_five, centroid_then_full, shape_then_full, primary_spot_then_full, boundary_safe_then_full.
selected_plan must equal the first plan in plan_ranking."""

USER_PREFIX = "Current fixed-gain meta-control state:\n"
