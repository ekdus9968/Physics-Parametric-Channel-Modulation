import cv2
import numpy as np
import os

# Jerlov IOP reference RGB ratios
# derived from beta_D(t): higher beta_D = more attenuation = lower ratio
JERLOV_REFS = {
    'I':   (0.18, 0.28, 0.54),  # blue dominant
    'II':  (0.22, 0.33, 0.45),
    'III': (0.28, 0.38, 0.34),  # green-ish
    '1C':  (0.30, 0.37, 0.33),
    '5C':  (0.33, 0.38, 0.29),
    '9C':  (0.36, 0.36, 0.28),  # yellow-brown
}

def estimate_water_type(img_path):
    """
    Estimate Jerlov water type from image RGB histogram.
    Returns water type string and distance scores.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Compute mean RGB values
    r = img[:,:,0].mean()
    g = img[:,:,1].mean()
    b = img[:,:,2].mean()
    total = r + g + b

    if total == 0:
        return 'III', {}

    # Normalize to ratios
    ratios = (r/total, g/total, b/total)

    # L2 distance to each Jerlov reference
    distances = {}
    for wtype, ref in JERLOV_REFS.items():
        dist = np.sqrt(sum((a-b)**2 for a, b in zip(ratios, ref)))
        distances[wtype] = dist

    # Closest match
    best_type = min(distances, key=distances.get)
    return best_type, distances

if __name__ == '__main__':
    BASE_PATH = 'data/S-UODAC2020'

    print("=" * 50)
    print("Phase 2: Water Type Estimator Validation")
    print("=" * 50)

    # Test on sample images from each type
    type_results = {}

    for type_name in ['type1','type2','type3','type4','type5','type6','type7']:
        type_path = os.path.join(BASE_PATH, type_name)
        if not os.path.exists(type_path):
            continue

        images = [f for f in os.listdir(type_path) if f.endswith('.jpg')][:20]

        predictions = []
        for fname in images:
            img_path = os.path.join(type_path, fname)
            wtype, _ = estimate_water_type(img_path)
            predictions.append(wtype)

        # Count distribution
        from collections import Counter
        dist = Counter(predictions)
        type_results[type_name] = dist
        print(f"\n{type_name} (20 samples):")
        for wt, count in sorted(dist.items()):
            bar = '█' * count
            print(f"  {wt:4s}: {bar} ({count})")