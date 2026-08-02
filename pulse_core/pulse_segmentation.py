import numpy as np

def segment_pulses(signal, onsets):
    """
    Segments the continuous PPG signal into individual cardiac cycles.
    Uses the foot (onset) of each pulse as the boundary.
    
    Args:
        signal (np.array): The filtered PPG signal.
        onsets (np.array): Indices of the pulse onsets.
        
    Returns:
        List of np.array, where each array is a single cardiac cycle.
        Also returns a list of starting indices for each segment.
    """
    segments = []
    segment_indices = []
    
    # We need at least 2 onsets to form 1 complete cycle
    if len(onsets) < 2:
        return segments, segment_indices
        
    for i in range(len(onsets) - 1):
        start_idx = onsets[i]
        end_idx = onsets[i+1]
        
        segment = signal[start_idx:end_idx]
        segments.append(segment)
        segment_indices.append(start_idx)
        
    return segments, segment_indices
