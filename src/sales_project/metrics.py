def mean_cv_scores(cv_res: dict, ndigits: int = 4) -> dict:
    """
    Calculates the mean of the cross-validation's scores in a dictionary
    for regression task.

    Args:
        cv_res (dict):
            A dictionary containing the cross-validation scores.
        ndigits (int):
            The number of decimal places to round the mean scores to.
            Defaults to 4.

    Returns:
        dict:
            A dictionary containing the mean cross-validation scores
            for regression task.
    """

    cv_res_mean = {}

    for key, value in cv_res.items():
        cv_res_mean[key] = round(value.mean(), ndigits)
        if "neg_" in key:
            cv_res_mean[key.replace("neg_", "")] = cv_res_mean[key] * -1
            del cv_res_mean[key]

    return cv_res_mean
