#!/usr/bin/env python3
"""
Shared utilities for the Missing Persons Outlier Detection project.
Single source of truth for state normalization, paths, and zone definitions.
"""
import os

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_DIR = "/home/yeblad/Desktop/Geospatial_Crime_Analysis"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
POP_DIR = os.path.join(DATA_DIR, "population")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
COVARIATES_DIR = os.path.join(DATA_DIR, "covariates")

for d in [DATA_DIR, RAW_DIR, POP_DIR, ANALYSIS_DIR, FEATURES_DIR, COVARIATES_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# State Abbreviation <-> Full Name Mappings
# =============================================================================
STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
    'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
    'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    # Territories
    'Puerto Rico': 'PR', 'Guam': 'GU', 'Virgin Islands': 'VI',
    'Northern Mariana Islands': 'MP', 'American Samoa': 'AS',
}

STATE_FULL = {v: k for k, v in STATE_ABBREV.items()}

# FIPS codes for states (used for county-level Census joins)
STATE_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
    'CO': '08', 'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12',
    'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
    'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23',
    'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
    'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38',
    'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'PR': '72',
    'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48',
    'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54',
    'WI': '55', 'WY': '56', 'GU': '66', 'VI': '78', 'MP': '69',
    'AS': '60',
}

FIPS_STATE = {v: k for k, v in STATE_FIPS.items()}


def normalize_state(value):
    """
    Normalize any state representation to its 2-letter abbreviation.

    Handles:
      - Already an abbreviation ('CA', 'TX')
      - Full name ('California', 'Texas')
      - Mixed case ('california', 'TEXAS')
      - Common variants ('Dist. of Columbia')

    Returns the abbreviation or the original value if unrecognized.
    """
    if not isinstance(value, str) or not value.strip():
        return value

    value = value.strip()

    # Already a known abbreviation
    upper = value.upper()
    if upper in STATE_FULL:
        return upper

    # Try full-name lookup (case-insensitive)
    lower = value.lower()
    for full_name, abbr in STATE_ABBREV.items():
        if full_name.lower() == lower:
            return abbr

    # Common variants (case-insensitive)
    variants = {
        'dist. of columbia': 'DC',
        'd.c.': 'DC',
        'us virgin islands': 'VI',
        'u.s. virgin islands': 'VI',
    }
    if lower in variants:
        return variants[lower]

    return value


def normalize_state_to_full(value):
    """
    Normalize any state representation to its full name.
    Returns the full name or the original value if unrecognized.
    """
    abbrev = normalize_state(value)
    return STATE_FULL.get(abbrev, value)


# =============================================================================
# Geographic Zone Definitions (single source of truth)
# =============================================================================
ZONES = {
    'US-Mexico Border': {
        'states': ['CA', 'AZ', 'NM', 'TX'],
        'counties': [
            'San Diego', 'Imperial', 'Yuma', 'Pima', 'Santa Cruz', 'Cochise',
            'Hidalgo', 'Luna', 'Dona Ana', 'El Paso', 'Hudspeth', 'Culberson',
            'Jeff Davis', 'Presidio', 'Brewster', 'Terrell', 'Val Verde',
            'Kinney', 'Maverick', 'Dimmit', 'Webb', 'Zapata', 'Starr',
            'Hidalgo', 'Cameron', 'Willacy', 'Brooks',
        ]
    },
    'I-35 Corridor': {
        'states': ['TX'],
        'counties': [
            'Denton', 'Collin', 'Dallas', 'Ellis', 'Hill', 'McLennan',
            'Bell', 'Williamson', 'Travis', 'Hays', 'Comal', 'Bexar',
            'Atascosa', 'Frio', 'Medina', 'Webb',
        ]
    },
    'Pacific Northwest': {
        'states': ['WA', 'OR'],
        'counties': [
            'King', 'Pierce', 'Snohomish', 'Thurston', 'Multnomah',
            'Clackamas', 'Washington', 'Marion',
        ]
    },
    'Midwest Metro': {
        'states': ['IL', 'WI', 'IN', 'OH', 'MI'],
        'counties': [
            'Cook', 'Milwaukee', 'Wayne', 'Marion', 'Cuyahoga',
            'Franklin', 'Hamilton',
        ]
    },
    'Northeast Corridor': {
        'states': ['NY', 'NJ', 'PA', 'MA', 'MD'],
        'counties': [
            'New York', 'Kings', 'Queens', 'Bronx', 'Richmond',
            'Philadelphia', 'Baltimore', 'Suffolk', 'Essex',
        ]
    },
    'Southern California': {
        'states': ['CA'],
        'counties': [
            'Los Angeles', 'Orange', 'San Diego', 'Riverside',
            'San Bernardino', 'Ventura',
        ]
    },
}

# US states only (excludes territories and regions)
US_STATE_ABBREVS = {
    abbrev for name, abbrev in STATE_ABBREV.items()
    if abbrev not in ('PR', 'GU', 'VI', 'MP', 'AS')
}
