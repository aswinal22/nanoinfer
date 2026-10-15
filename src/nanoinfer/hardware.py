"""Accelerator specs used by the roofline model.

Numbers here are *vendor datasheet peaks with sparsity disabled*. Real inference
never approaches them. A well-tuned dense GEMM on a modern accelerator lands
around 50-80% of peak; a decode step, which is dominated by skinny GEMVs,
lands far lower. Treat these as ceilings, never as expected throughput.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    """Peak capabilities of an accelerator.

    Attributes:
        name: Human-readable identifier.
        peak_tflops: Peak dense tensor-core throughput in TFLOP/s for `dtype`.
        bandwidth_gbs: Peak HBM/GDDR bandwidth in GB/s.
        dtype: The numeric format the peaks are quoted for.
        source: Where the numbers come from, for provenance.
    """

    name: str
    peak_tflops: float
    bandwidth_gbs: float
    dtype: str = "float16"
    source: str = ""

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) at which compute overtakes memory.

        Above this intensity a kernel is compute-bound; below it, bandwidth-bound.
        This single number predicts the behaviour of every kernel on the device.
        """
        return self.peak_tflops * 1e12 / (self.bandwidth_gbs * 1e9)

    def describe(self) -> str:
        return (
            f"{self.name:<14} {self.peak_tflops:>7.1f} TFLOP/s  "
            f"{self.bandwidth_gbs:>6.0f} GB/s  ridge={self.ridge_point:>6.1f} FLOP/byte"
        )


# --- Datacenter ------------------------------------------------------------
A100_40GB = Hardware("A100-40GB", 312.0, 1555.0, source="NVIDIA A100 datasheet")
A100_80GB = Hardware("A100-80GB", 312.0, 2039.0, source="NVIDIA A100 datasheet")
H100_SXM = Hardware("H100-SXM", 989.0, 3350.0, source="NVIDIA H100 datasheet")
H100_PCIe = Hardware("H100-PCIe", 756.0, 2600.0, source="NVIDIA H100 datasheet")
MI300X = Hardware("MI300X", 1307.0, 5300.0, source="AMD MI300X datasheet")

# --- Inference / edge ------------------------------------------------------
L4 = Hardware("L4", 121.0, 300.0, source="NVIDIA L4 datasheet")
A10G = Hardware("A10G", 125.0, 600.0, source="NVIDIA A10 datasheet")
T4 = Hardware("T4", 65.0, 320.0, source="NVIDIA T4 datasheet")
RTX_4090 = Hardware("RTX-4090", 165.0, 1008.0, source="NVIDIA Ada whitepaper")

ALL: tuple[Hardware, ...] = (
    A100_40GB,
    A100_80GB,
    H100_SXM,
    H100_PCIe,
    MI300X,
    L4,
    A10G,
    T4,
    RTX_4090,
)

BY_NAME: dict[str, Hardware] = {hw.name: hw for hw in ALL}


def get(name: str) -> Hardware:
    """Look up a hardware spec by name (case-insensitive, tolerant of spacing)."""
    key = name.strip().upper().replace(" ", "").replace("_", "").replace("-", "")
    for hw in ALL:
        candidate = hw.name.strip().upper().replace(" ", "").replace("_", "").replace("-", "")
        if candidate == key:
            return hw
    raise KeyError(f"Unknown hardware {name!r}. Known: {sorted(BY_NAME)}")