from dataclasses import dataclass, field
import numpy as np

@dataclass
class DetectionResult:
    """
    Standardized output from any peak detector in the platform.
    """
    systolic_peaks: np.ndarray = field(default_factory=lambda: np.array([]))
    onsets: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence: float = 0.0
    detector_used: str = "none"
    morphology_available: bool = False
    sqi_score: float = 0.0
    rejection_reason: str = "None"
    fiducials: dict = field(default_factory=dict)
    
class BaseDetector:
    """
    Interface for all peak detectors.
    """
    def detect(self, signal: np.ndarray, fs: float, sqi_score: float = 0.0) -> DetectionResult:
        raise NotImplementedError("Subclasses must implement detect()")
