# Configuration file for physiological sensing pipeline

# Signal Quality Index (SQI) Thresholds
# These govern the dynamic routing of the signal to different peak detectors.
# Spectral Power Ratio (SPR) is the ratio of power in the 0.5Hz - 3.0Hz band.
SQI_GOOD_THRESHOLD = 0.60
SQI_MODERATE_THRESHOLD = 0.30

# Feature Extraction Confidence Thresholds
# Defines which features are reliable given the detector's confidence score.
CONFIDENCE_HIGH = 0.90
CONFIDENCE_MEDIUM = 0.60
