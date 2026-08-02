import numpy as np

# ayurveda_mapper.py

def map_to_ayurveda(scientific_features, baseline=None, sigma_multiplier=1.0):
    """
    Maps scientifically validated hemodynamic and morphological features
    to classical Ayurvedic Nadi Pariksha descriptors, and computes a dominant Dosha.
    If a baseline is provided, uses dynamic statistical thresholding for AI and h1.
    """
    ayurveda_profile = {}
    
    vata_score = 0
    pitta_score = 0
    kapha_score = 0
    
    # 1. VEGA (Pulse Rate)
    # Vata: 80 - 95 bpm, Pitta: 70 - 80 bpm, Kapha: 50 - 60 bpm
    if 'Heart_Rate_BPM' in scientific_features:
        hr = scientific_features['Heart_Rate_BPM']
        ayurveda_profile['Vega_HeartRate_BPM'] = hr
        if hr >= 80:
            vata_score += 1
            ayurveda_profile['Vega_Dosha'] = 'Vata (Fast)'
        elif hr >= 70:
            pitta_score += 1
            ayurveda_profile['Vega_Dosha'] = 'Pitta (Medium)'
        else:
            kapha_score += 1
            ayurveda_profile['Vega_Dosha'] = 'Kapha (Slow)'

    # 2. GATI (Movement / Wave Shape)
    # Reflected in T2 (Peak-to-Peak interval) and its interaction with amplitude/width.
    if 'Heart_Rate_BPM' in scientific_features:
        hr = scientific_features['Heart_Rate_BPM']
        if hr >= 80:
            vata_score += 1
            ayurveda_profile['Gati_Dosha'] = 'Vata (Curved/Zigzag - Snake)'
        elif hr >= 70:
            pitta_score += 1
            ayurveda_profile['Gati_Dosha'] = 'Pitta (Hopping/Jumping - Frog)'
        else:
            kapha_score += 1
            ayurveda_profile['Gati_Dosha'] = 'Kapha (Slow/Broad - Swan)'

    # 3. TALA (Rhythm and Stability)
    if 'CVNN' in scientific_features:
        cvnn = scientific_features['CVNN']
        ayurveda_profile['Tala_CVNN'] = cvnn
        if cvnn > 5.0:
            vata_score += 1
            ayurveda_profile['Tala_Dosha'] = 'Vata (Irregular)'
        else:
            pitta_score += 0.5
            kapha_score += 0.5
            ayurveda_profile['Tala_Dosha'] = 'Pitta/Kapha (Regular)'

    # 4. BALA (Force)
    # AI = (h1 - h3) / h1.
    if 'Augmentation_Index' in scientific_features and not np.isnan(scientific_features['Augmentation_Index']):
        ai = scientific_features['Augmentation_Index']
        ayurveda_profile['Bala_Augmentation_Index'] = ai
        
        if baseline and 'ai_median' in baseline and 'ai_std' in baseline:
            # Dynamic Thresholding
            med = baseline['ai_median']
            std = baseline['ai_std'] * sigma_multiplier
            
            if ai > (med + std):
                pitta_score += 1
                ayurveda_profile['Bala_Dosha'] = 'Pitta (High Force)'
            elif ai < (med - std):
                vata_score += 1
                ayurveda_profile['Bala_Dosha'] = 'Vata (Low Force)'
            else:
                kapha_score += 1
                ayurveda_profile['Bala_Dosha'] = 'Kapha (Moderate Force)'
        else:
            # Fallback static thresholds
            if ai > 0.15:
                pitta_score += 1
                ayurveda_profile['Bala_Dosha'] = 'Pitta (High Force)'
            elif ai >= 0.0:
                kapha_score += 1
                ayurveda_profile['Bala_Dosha'] = 'Kapha (Moderate Force)'
            else:
                vata_score += 1
                ayurveda_profile['Bala_Dosha'] = 'Vata (Low Force)'

    # 5. KATHINYA (Stiffness and Consistency)
    if 'Stiffness_Index' in scientific_features and not np.isnan(scientific_features['Stiffness_Index']):
        si = scientific_features['Stiffness_Index']
        ayurveda_profile['Kathinya_Stiffness_Index'] = si
        if si > 10:
            vata_score += 1
            ayurveda_profile['Kathinya_Dosha'] = 'Vata (Hard/Rough)'
        elif si > 6:
            kapha_score += 1
            ayurveda_profile['Kathinya_Dosha'] = 'Kapha (Soft Thickening)'
        else:
            pitta_score += 1
            ayurveda_profile['Kathinya_Dosha'] = 'Pitta (Elastic/Flexible)'

    # 6. AKRUTI (Volume and Tension)
    if 'h1_Amplitude_P' in scientific_features and not np.isnan(scientific_features['h1_Amplitude_P']):
        h1 = scientific_features['h1_Amplitude_P']
        ayurveda_profile['Akruti_h1'] = h1
        
        if baseline and 'h1_median' in baseline and 'h1_std' in baseline:
            # Dynamic Thresholding
            med = baseline['h1_median']
            std = baseline['h1_std'] * sigma_multiplier
            # Ensure std isn't 0
            if std == 0: std = med * 0.05
            
            if h1 > (med + std):
                pitta_score += 1
                ayurveda_profile['Akruti_Dosha'] = 'Pitta (High Volume)'
            elif h1 < (med - std):
                vata_score += 1
                ayurveda_profile['Akruti_Dosha'] = 'Vata (Low Volume)'
            else:
                kapha_score += 1
                ayurveda_profile['Akruti_Dosha'] = 'Kapha (Average Volume)'
        else:
            # Fallback static thresholds
            if h1 < 5000: 
                vata_score += 1
                ayurveda_profile['Akruti_Dosha'] = 'Vata (Low Volume)'
            elif h1 < 20000:
                kapha_score += 1
                ayurveda_profile['Akruti_Dosha'] = 'Kapha (Average Volume)'
            else:
                pitta_score += 1
                ayurveda_profile['Akruti_Dosha'] = 'Pitta (High Volume)'

    ayurveda_profile['Scores'] = {
        'Vata': vata_score,
        'Pitta': pitta_score,
        'Kapha': kapha_score
    }
    
    # Determine dominant Dosha and handle ties
    scores = {'Vata': vata_score, 'Pitta': pitta_score, 'Kapha': kapha_score}
    max_score = max(scores.values())
    dominant_doshas = [dosha for dosha, score in scores.items() if score == max_score]
    
    if len(dominant_doshas) > 1:
        dominant_dosha = "-".join(dominant_doshas) + " (Tie)"
    else:
        dominant_dosha = dominant_doshas[0]
        
    ayurveda_profile['Dominant_Dosha'] = dominant_dosha

    return ayurveda_profile
