import numpy as np

def get_covariance_correction(n_s, n_d, n_theta=None, method='percival'):
    """
    Correction factor to debias de inverse covariance matrix.

    Args:
        n_s (int): Number of simulations.
        n_d (int): Number of bins of the data vector.
        n_theta (int): Number of free parameters.
        method (str): Method to compute the correction factor.

    Returns:
        float: Correction factor
    """
    if method == 'percival':
        B = (n_s - n_d - 2) / ((n_s - n_d - 1)*(n_s - n_d - 4))
        return (n_s - 1)*(1 + B*(n_d - n_theta))/(n_s - n_d + n_theta - 1)
    elif method == 'percival-fisher':
        return (n_s - 1)/(n_s - n_d + n_theta - 1)
    elif method == 'hartlap':
        return (n_s - 1)/(n_s - n_d - 2)
    else:
        raise ValueError(f"Unknown method: {method}. Available methods are: 'percival', 'percival-fisher', 'hartlap'.")

def correlation_from_covariance(covariance):
    """
    Compute the correlation matrix from the covariance matrix.

    Parameters
    ----------
    covariance : array_like
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Correlation matrix.
    """
    
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def is_camel_case(s: str) -> bool:
    """
    Check if a string is in camel case.

    Parameters
    ----------
    s : str
        String to check.

    Returns
    -------
    bool
        True if the string is in camel case, False otherwise.
    """
    return s != s.lower() and s != s.upper() and "_" not in s

def project_observable_names(project):
    """
    Get the names of all observables in a project module by 
    checking the attributes of the module and selecting the 
    ones that are CamelCase and have a stat_name attribute.
    
    Parameters
    ----------
    project : module
        The project module to inspect.
        
    Returns
    -------
    list[str]
        A sorted list of observable names.
    """
    observables_names = [
        getattr(project, attr).__name__ 
        for attr in dir(project) # get all attributes of the module
        if is_camel_case(attr) # Ensure only the classes are called (CamelCase by convention)
        and hasattr(getattr(project, attr), 'stat_name') # Ensure it is an observable class
    ]
    observables_names.sort()
    return observables_names