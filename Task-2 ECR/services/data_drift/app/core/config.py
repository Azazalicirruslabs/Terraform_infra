ALLOWED_ANALYSIS_TYPES = {
    "Data Drift": "datadrift",
    "Fairness": "fairness",
    "Classification": "classification",
    "Regression": "regression",
}


def check_analysis(analysis_type):

    analysis = ""
    for key, val in ALLOWED_ANALYSIS_TYPES.items():
        if analysis_type in key:
            analysis = val
        else:
            continue
    return analysis
