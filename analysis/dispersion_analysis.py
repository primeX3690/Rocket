"""
analysis/dispersion_analysis.py

Circular Error Probable (CEP), dispersion ellipses, and parameter-
sensitivity data for Monte Carlo insertion-error results (from
analysis/monte_carlo.py) -- the standard way mission-analysis teams
summarize a 2D scatter of miss-distances into a few numbers/shapes a
non-specialist can read at a glance, rather than a raw point cloud.

CEP is computed via the genuine statistical definition (the radius of
the circle containing 50% of the scatter, found from the actual
empirical distribution -- not the common but only-valid-for-circular-
Gaussian shortcut formula, which this module deliberately avoids so it
stays correct even for elongated (non-circular) dispersion).

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np


def circular_error_probable(x_errors: np.ndarray, y_errors: np.ndarray, percentile: float = 50.0) -> float:
    """
    CEP (or a generalized "CEP_p" for any percentile): the radius R such
    that `percentile`% of the (x,y) miss points fall within R of the
    mean impact point. Computed directly from the empirical radial-
    distance distribution -- exact for any scatter shape, not just a
    circular Gaussian.
    """
    x_errors, y_errors = np.asarray(x_errors), np.asarray(y_errors)
    mean_x, mean_y = np.mean(x_errors), np.mean(y_errors)
    radial = np.sqrt((x_errors - mean_x) ** 2 + (y_errors - mean_y) ** 2)
    return float(np.percentile(radial, percentile))


def dispersion_ellipse(x_errors: np.ndarray, y_errors: np.ndarray, confidence: float = 0.95):
    """
    The confidence-level dispersion ellipse from the scatter's own
    covariance matrix: eigenvectors give the ellipse's principal axes
    (the directions of maximum/minimum spread -- NOT necessarily
    aligned with x/y, e.g. a PEG burn's downrange vs cross-range error
    are rarely equal), eigenvalues give the axis lengths (scaled by the
    chi-square quantile for the requested confidence level, the
    standard construction for a bivariate-Gaussian confidence region).
    """
    from scipy.stats import chi2
    x_errors, y_errors = np.asarray(x_errors), np.asarray(y_errors)
    mean = np.array([np.mean(x_errors), np.mean(y_errors)])
    cov = np.cov(x_errors, y_errors)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    scale = np.sqrt(chi2.ppf(confidence, df=2))
    semi_major = scale * np.sqrt(max(eigenvalues[0], 0.0))
    semi_minor = scale * np.sqrt(max(eigenvalues[1], 0.0))
    angle_rad = float(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_local = np.array([semi_major * np.cos(theta), semi_minor * np.sin(theta)])
    rotation = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    ellipse_points = (rotation @ ellipse_local).T + mean

    return {"center": mean, "semi_major_m": semi_major, "semi_minor_m": semi_minor,
            "angle_rad": angle_rad, "confidence": confidence, "boundary_points": ellipse_points}


def sensitivity_heatmap_data(monte_carlo_result: dict, dispersed_params: dict) -> dict:
    """
    Parameter-sensitivity data: Pearson correlation between each
    dispersed input parameter and the resulting radius/velocity error,
    the standard first-pass "which input matters most" screening used
    before a full Sobol/variance-based sensitivity study. Returns a
    matrix suitable for a heatmap (params x output metrics).
    """
    outputs = {
        "radius_error_m": np.asarray(monte_carlo_result["radius_error_m"]),
        "velocity_error_m_s": np.asarray(monte_carlo_result["velocity_error_m_s"]),
    }
    matrix = np.zeros((len(dispersed_params), len(outputs)))
    param_names = list(dispersed_params.keys())
    output_names = list(outputs.keys())
    for i, pname in enumerate(param_names):
        pvals = np.asarray(dispersed_params[pname])
        for j, oname in enumerate(output_names):
            if np.std(pvals) < 1e-12 or np.std(outputs[oname]) < 1e-12:
                matrix[i, j] = 0.0
            else:
                matrix[i, j] = np.corrcoef(pvals, outputs[oname])[0, 1]
    return {"param_names": param_names, "output_names": output_names, "correlation_matrix": matrix}
