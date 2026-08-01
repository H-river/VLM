"""Typed orchestration failures."""


class OrchestrationError(RuntimeError):
    """Base class for failures safe to expose to the caller."""


class ContractError(OrchestrationError):
    """The proposed orchestration decision violates the frozen contract."""


class SpecialistError(OrchestrationError):
    """A validated specialist call could not be completed."""

