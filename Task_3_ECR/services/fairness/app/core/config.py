THEIL_INDEX_MULTIPLIER = 200
DIFFERENCE_THRESHOLD = 0.1
POINTS_PENALTY_PER_THRESHOLD = 20

ALLOWED_ANALYSIS_TYPES = {"Data Drift": "datadrift", "Fairness": "fairness", "Classification Stats": "classification"}


def check_analysis(analysis_type):

    analysis = ""
    for key, val in ALLOWED_ANALYSIS_TYPES.items():
        if analysis_type in key:
            analysis = val
        else:
            continue
    return analysis
