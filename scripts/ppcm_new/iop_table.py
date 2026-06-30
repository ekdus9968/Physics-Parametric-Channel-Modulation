# IOP Table - Solonenko & Mobley, Applied Optics 2015
# beta_D: direct attenuation coefficient per channel (m^-1)
# beta_B: backscatter attenuation coefficient per channel (m^-1)
# B_inf:  background (veiling) light per channel [0,1]

BETA_D = {
    'I':   {'R': 0.345, 'G': 0.073, 'B': 0.017},
    'II':  {'R': 0.179, 'G': 0.082, 'B': 0.024},
    'III': {'R': 0.135, 'G': 0.089, 'B': 0.038},
    '1C':  {'R': 0.179, 'G': 0.082, 'B': 0.047},
    '5C':  {'R': 0.245, 'G': 0.156, 'B': 0.245},
    '9C':  {'R': 0.290, 'G': 0.199, 'B': 0.349},
}

# Backscatter attenuation (approximately 0.8 * beta_D based on literature)
BETA_B = {
    wt: {c: v * 0.8 for c, v in beta.items()}
    for wt, beta in BETA_D.items()
}

# Background (veiling) light per channel
B_INF = {
    'I':   {'R': 0.12, 'G': 0.08, 'B': 0.15},
    'II':  {'R': 0.14, 'G': 0.10, 'B': 0.13},
    'III': {'R': 0.16, 'G': 0.12, 'B': 0.12},
    '1C':  {'R': 0.17, 'G': 0.13, 'B': 0.11},
    '5C':  {'R': 0.19, 'G': 0.15, 'B': 0.10},
    '9C':  {'R': 0.22, 'G': 0.18, 'B': 0.09},
}

# Depth range assumption for relative → absolute scaling
DEPTH_MIN = 0.5   # meters
DEPTH_MAX  = 10.0 # meters

ALL_WATER_TYPES = ['I', 'II', 'III', '1C', '5C', '9C']

def get_dominant_beta(water_type):
    """Mean of R,G,B beta_D for spatial weighting."""
    beta = BETA_D[water_type]
    return (beta['R'] + beta['G'] + beta['B']) / 3.0