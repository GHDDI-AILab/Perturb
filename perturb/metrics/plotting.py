__all__ = [
    'kl_divergence',
    'js_divergence',
    'plot_corr_matrix',
]

import numpy as np
import matplotlib.pyplot as plt

def kl_divergence(p, q):
    epsilon = 1e-10 
    p = p + epsilon
    q = q + epsilon
    return np.sum(p * np.log(p / q))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def plot_corr_matrix(
        split_target: list[np.ndarray],
        split_output: list[np.ndarray],
        unique_smiles: list[str],
        save_path: str = 'CorrHeatmaps.png'
    ) -> dict[str, np.ndarray]:
    """
    Compute and visualize the correlation matrices of target and output values.

    Parameters:
    - split_target: List of target data splits.
    - split_output: List of model output data splits.
    - unique_smiles: List of unique SMILES strings representing compounds.
    - save_path: Path to save the image.

    Returns:
    - A dict with the target and output correlation matrices.
    """
    targets_corr_matrix = np.corrcoef(
        np.stack([np.mean(a, axis=0) for a in split_target])
    )
    outputs_corr_matrix = np.corrcoef(
        np.stack([np.mean(a, axis=0) for a in split_output])
    )

    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))

    cax1 = axarr[0].matshow(targets_corr_matrix, cmap='RdBu_r')
    axarr[0].set_title('Intra-Perturbation Correlation Matrix')
    axarr[0].set_ylabel('Test Compounds')
    plt.colorbar(cax1, ax=axarr[0])

    cax2 = axarr[1].matshow(outputs_corr_matrix, cmap='RdBu_r')
    axarr[1].set_title('Intra-Prediction Correlation Matrix')
    plt.colorbar(cax2, ax=axarr[1])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return {
        "targets_corr_matrix": targets_corr_matrix,
        "outputs_corr_matrix": outputs_corr_matrix,
        "fig": fig,
    }

