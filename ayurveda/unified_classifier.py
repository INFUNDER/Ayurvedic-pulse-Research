import numpy as np

class UnifiedClassifier:
    """
    Evaluates the spatial contrast between the three independent Nadi Pariksha sensors
    (Vata/Index, Pitta/Middle, Kapha/Ring) to generate a unified Ayurvedic diagnostic.
    """
    
    def __init__(self):
        self.spatial_weight = 1.5 # Reward Dosha scores that align with their anatomical finger
        
    def classify(self, vata_report, pitta_report, kapha_report):
        """
        Takes the three independent Dosha reports (dicts from map_to_ayurveda)
        Returns a unified diagnostic dict.
        If a sensor failed (Poor SQI / No Beats), its report should be None.
        """
        
        valid_sensors = 0
        total_vata = 0.0
        total_pitta = 0.0
        total_kapha = 0.0
        heart_rates = []
        
        # 1. Process Vata Sensor (Index Finger)
        if vata_report:
            valid_sensors += 1
            scores = vata_report.get('Scores', {})
            total_vata += scores.get('Vata', 0) * self.spatial_weight
            total_pitta += scores.get('Pitta', 0)
            total_kapha += scores.get('Kapha', 0)
            if not np.isnan(vata_report.get('Vega_HeartRate_BPM', np.nan)):
                heart_rates.append(vata_report['Vega_HeartRate_BPM'])
                
        # 2. Process Pitta Sensor (Middle Finger)
        if pitta_report:
            valid_sensors += 1
            scores = pitta_report.get('Scores', {})
            total_vata += scores.get('Vata', 0)
            total_pitta += scores.get('Pitta', 0) * self.spatial_weight
            total_kapha += scores.get('Kapha', 0)
            if not np.isnan(pitta_report.get('Vega_HeartRate_BPM', np.nan)):
                heart_rates.append(pitta_report['Vega_HeartRate_BPM'])
                
        # 3. Process Kapha Sensor (Ring Finger)
        if kapha_report:
            valid_sensors += 1
            scores = kapha_report.get('Scores', {})
            total_vata += scores.get('Vata', 0)
            total_pitta += scores.get('Pitta', 0)
            total_kapha += scores.get('Kapha', 0) * self.spatial_weight
            if not np.isnan(kapha_report.get('Vega_HeartRate_BPM', np.nan)):
                heart_rates.append(kapha_report['Vega_HeartRate_BPM'])
                
        # Handle case where all sensors failed
        if valid_sensors == 0:
            return {
                "Primary_Dosha": "N/A",
                "Secondary_Dosha": "N/A",
                "Confidence": 0,
                "Reason": "All sensors failed quality checks or beat detection."
            }
            
        # Determine Dominant Dosha(s)
        final_scores = {
            "Vata": total_vata,
            "Pitta": total_pitta,
            "Kapha": total_kapha
        }
        
        sorted_doshas = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
        
        primary_dosha_name, primary_score = sorted_doshas[0]
        secondary_dosha_name, secondary_score = sorted_doshas[1]
        
        # Check for tie/combined Dosha (if secondary is within 1.0 points of primary)
        is_combined = False
        if (primary_score - secondary_score) <= 1.0 and secondary_score > 0:
            is_combined = True
            
        # Calculate Confidence Score (0-100%)
        confidence = 100
        
        # Penalty for failed sensors
        if valid_sensors == 2:
            confidence -= 25
        elif valid_sensors == 1:
            confidence -= 50
            
        # Penalty for heart rate disagreement (indicates motion artifact on one finger)
        if len(heart_rates) >= 2:
            hr_spread = max(heart_rates) - min(heart_rates)
            if hr_spread > 15:
                confidence -= 20
            elif hr_spread > 8:
                confidence -= 10
                
        # Penalty for weak signal (primary score is very low)
        max_possible_score = valid_sensors * 5 * self.spatial_weight
        if primary_score < (max_possible_score * 0.3):
            confidence -= 15
            
        confidence = int(max(0, min(100, confidence))) # Clamp between 0 and 100
        
        return {
            "Primary_Dosha": f"{primary_dosha_name}-{secondary_dosha_name}" if is_combined else primary_dosha_name,
            "Secondary_Dosha": secondary_dosha_name if not is_combined else sorted_doshas[2][0],
            "Confidence": confidence,
            "Scores": final_scores,
            "Valid_Sensors": valid_sensors
        }
