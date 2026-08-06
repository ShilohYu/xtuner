"""Configuration contract for Xtuner's optional UltraEP execution path."""

from __future__ import annotations

from typing import Annotated, Literal

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, model_validator


class UltraEPConfig(BaseModel):
    """Runtime-only redundant-expert configuration.

    ``MoEConfig.ultraep_cfg is None`` is the only disabled state. Replica weights
    and gradients remain owned by the UltraEP runtime; this configuration never
    changes model parameters, optimizer state, or checkpoints.
    """

    model_config = ConfigDict(extra="forbid")

    num_redundant_experts_per_rank: Annotated[
        int, Parameter(help="UltraEP redundant-expert slots reserved on each EP rank")
    ]

    @model_validator(mode="after")
    def validate_redundant_experts(self) -> "UltraEPConfig":
        if self.num_redundant_experts_per_rank <= 0:
            raise ValueError("num_redundant_experts_per_rank must be > 0")
        return self


def validate_ultraep_fsdp_compatibility(
    ultraep: UltraEPConfig | None,
    *,
    recompute_ratio: float,
) -> None:
    """Reject an execution mode whose backward ordering UltraEP cannot support."""
    if ultraep is not None and recompute_ratio > 0:
        raise ValueError(
            "Xtuner UltraEP does not support FSDP activation recompute yet; "
            "set fsdp_config.recompute_ratio=0 or disable ultraep"
        )


def validate_ultraep_moe_compatibility(
    ultraep: UltraEPConfig | None,
    *,
    ep_size: int,
    n_routed_experts: int,
    dispatcher: Literal["deepep", "all2all", "agrs"] | None,
    float8_enabled: bool,
    moe_bias: bool,
) -> None:
    """Validate UltraEP against the MoE execution path that will be built."""
    if ultraep is None:
        return
    if ep_size <= 1:
        raise ValueError("UltraEP requires ep_size > 1")
    if n_routed_experts % ep_size != 0:
        raise ValueError("UltraEP requires n_routed_experts to be divisible by ep_size")
    if dispatcher != "deepep":
        raise ValueError("Xtuner UltraEP currently supports dispatcher='deepep' only")
    if float8_enabled:
        raise ValueError("Xtuner UltraEP currently supports BF16 grouped experts only")
    if moe_bias:
        raise ValueError("Xtuner UltraEP does not currently support expert bias")


def validate_ultraep_mtp_compatibility(
    ultraep: UltraEPConfig | None,
    *,
    mtp_enabled: bool,
) -> None:
    """Reject MTP expert layers until their UltraEP lifecycle is supported."""
    if ultraep is not None and mtp_enabled:
        raise ValueError("Xtuner UltraEP does not currently support MTP expert layers")
