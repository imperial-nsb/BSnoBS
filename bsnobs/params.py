from dataclasses import dataclass


@dataclass
class AnalysisParameters:
    # Microscope
    scale_um_per_pixel: float = 0.0825
    sample_volume_per_frame_uL: float = 0.00089

    # Size filtering
    min_diameter_um: float = 0.5
    max_diameter_um: float = 20.0

    # Histogram
    bin_size_um: float = 0.165

    # Gas dose calculation
    injection_volume_uL: float = 100.0

    # Model
    pretrained_model: str = ""
