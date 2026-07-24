import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def calculate_one_vs_rest_density_ratio_scores(X_train, y_train, X_test, test_indices, classes, d, norm, k, eps, n_permutations, group_keys, base_seed=42):
    """
    This function computes the density ratio scores for each class.

    Parameters:
    ----------
    X_train: np.ndarray
        The training data, shape (n_train_samples, n_features).
    y_train: np.ndarray
        The training labels, shape (n_train_samples,).
    X_test: np.ndarray
        The test data, shape (n_test_samples, n_features).
    test_indices: pd.MultiIndex
        The indices of the test samples, shape (n_test_samples,).
    classes: list
        The unique classes in the training labels.
    d: int
        The dimensionality of the data.
    norm: bool
        Whether to normalize the density ratio scores using a null distribution.
    k: int
        The number of nearest neighbors to use in the density ratio score calculation.
    eps: float
        A small value to avoid division by zero in the density ratio score calculation.
    n_permutations: int
        The number of permutations to perform for the null distribution.
    group_keys: list
        The keys to group by when calculating the mean and std of the null scores. For example, ['chunk_idx', 'scheme_idx', 'modulation']. Basically, the keys that uniquely identify each test sample.
    base_seed: int, optional
        The base seed for the random number generator to ensure reproducibility. Default is 42.

    Returns:
    -------
    score_df: pd.DataFrame
        A DataFrame containing the density ratio scores for each class, shape (n_test_samples, n_classes).
    """
    score_df = calculate_one_vs_rest_density_ratio_scores_helper(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        test_indices=test_indices,
        classes=classes,
        d=d,
        k=k,
        eps=eps,
    )

    if norm:
        null_mean, null_std = calculate_null_distribution_density_ratio_scores(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            test_indices=test_indices,
            classes=classes,
            d=d,
            k=k,
            eps=eps,
            n_permutations=n_permutations,
            group_keys=group_keys,
            base_seed=base_seed,
        )

        score_df = normalize_density_ratio_scores(
            score_df,
            null_mean,
            null_std,
        )

    return score_df

def calculate_null_distribution_density_ratio_scores(X_train, y_train, X_test, test_indices, classes, d, k, eps, n_permutations, group_keys, base_seed=42):
    '''
    This function computes density ratio scores across permutations of the labels to create a null distribution of scores. This will be used to normalize the actual scores later.

    Parameters:
    ----------
    X_train: np.ndarray
        The training data, shape (n_train_samples, n_features).
    y_train: np.ndarray
        The training labels, shape (n_train_samples,).
    X_test: np.ndarray
        The test data, shape (n_test_samples, n_features).
    test_indices: pd.MultiIndex
        The indices of the test samples, shape (n_test_samples,).
    classes: list
        The unique classes in the training labels.
    d: int
        The dimensionality of the data.
    k: int
        The number of nearest neighbors to use in the density ratio score calculation.
    eps: float
        A small value to avoid division by zero in the density ratio score calculation.
    n_permutations: int
        The number of permutations to perform for the null distribution.
    group_keys: list
        The keys to group by when calculating the mean and std of the null scores. For example, ['chunk_idx', 'scheme_idx', 'modulation']. Basically, the keys that uniquely identify each test sample.
    base_seed: int, optional
        The base seed for the random number generator to ensure reproducibility. Default is 42.
    '''
    assert X_train.shape[0] == y_train.shape[0], "The number of training samples must match the number of training labels."
    assert X_test.shape[0] == test_indices.shape[0], "The number of test samples must match the number of test indices."
    assert len(classes) == len(np.unique(y_train)), "The number of unique classes must match the number of unique training labels."

    # Compute density ratio scores across permutations of the modulation labels to create a null distribution of scores. This will be used to normalize the actual scores later.
    # NOTE: I don't know if the permutation stuff will work with UMAP, because UMAP takes into account the labels and those change for each permutation. 
    score_df_permutations = []
    for perm in range(n_permutations):
        print(f"Calculating null distribution for permutation {perm + 1}/{n_permutations}...")
        # Randomly permute the training modulation labels to generate a null distribution.
        rng = np.random.default_rng(seed=base_seed + perm)  # For reproducibility
        y_perm = y_train.copy()
        rng.shuffle(y_perm)

        score_df_per_class = {}

        for cls in classes:

            in_mask = (y_perm == cls)
            out_mask = ~in_mask

            X_ref = X_train[in_mask]
            Y_ref = X_train[out_mask]

            s_fold = density_ratio_score(
                    X_query=X_test,
                    X_ref=X_ref,
                    Y_ref=Y_ref,
                    d=d,
                    k=k,
                    eps=eps,
                )

            df_cls = pd.DataFrame(
                s_fold, 
                index=test_indices, 
                columns=[cls]
            )

            score_df_per_class[cls] = df_cls
        # Concat score_df_per_class into a single DataFrame
        score_df = pd.concat(score_df_per_class.values(), axis=1)
        score_df['permutation'] = perm
        score_df.set_index('permutation', append=True, inplace=True)
        score_df_permutations.append(score_df)
    null_scores = pd.concat(score_df_permutations) # Shape: (n_permutations * n_test_samples, n_classes)
    null_scores = null_scores[classes] # Reorder the columns by classes

    # Take the mean and std of the null scores for each test sample and class across permutations. This will be used to normalize the actual scores later.
    null_mean = null_scores.groupby(level=group_keys).mean() # Shape: (n_test_samples, n_classes)
    null_std = null_scores.groupby(level=group_keys).std() # Shape: (n_test_samples, n_classes)

    return null_mean, null_std

def normalize_density_ratio_scores(score_df, null_mean, null_std):
    '''
    This function normalizes the density ratio scores using the null hypothesis mean and std generated by the calculate_null_distribution_density_ratio_scores() function.

    Assertions ensure that the indices and columns of the score_df, null_mean, and null_std DataFrames match before performing the normalization.

    Parameters:
    ----------
    score_df: pd.DataFrame
        The density ratio scores for the test samples, shape (n_test_samples, n_classes).
    null_mean: pd.DataFrame
        The mean of the null distribution of density ratio scores for the test samples, shape (n_test_samples, n_classes).
    null_std: pd.DataFrame
        The std of the null distribution of density ratio scores for the test samples, shape (n_test_samples, n_classes).

    Returns:
    -------
    score_df: pd.DataFrame
        The normalized density ratio scores for the test samples, shape (n_test_samples, n_classes).
    '''
    assert (score_df.index == null_mean.index).all()
    assert (score_df.index == null_std.index).all()
    assert (score_df.columns == null_mean.columns).all()
    assert (score_df.columns == null_std.columns).all()
    score_df = (score_df - null_mean) / (null_std + 1e-12)
    return score_df

def calculate_one_vs_rest_density_ratio_scores_helper(X_train, y_train, X_test, test_indices, classes, d, k=4, eps=1e-7):
    '''
    This function computes the density ratio scores for each class in a one-vs-rest manner. 
    For each class, it calculates the density ratio score of the test samples against the training samples 
    of that class (in-class) and the training samples of all other classes (out-of-class).

    Parameters:
    ----------
    X_train: np.ndarray
        The training data, shape (n_train_samples, n_features).
    y_train: np.ndarray
        The training labels, shape (n_train_samples,).
    X_test: np.ndarray
        The test data, shape (n_test_samples, n_features). These are the samples for which we want to compute the density ratio scores for.
    test_indices: pd.MultiIndex
        The indices of the test samples, shape (n_test_samples,).
    classes: list
        The unique classes in the training labels.
    d: int
        The dimensionality of the data.
    k: int, optional
        The number of nearest neighbors to use in the density ratio score calculation. Default is 4.
    eps: float, optional
        A small value to avoid division by zero in the density ratio score calculation. Default is 1e-7.

    Returns:
    -------
    score_df: pd.DataFrame
        A DataFrame containing the density ratio scores for each class, shape (n_test_samples, n_classes).
    '''
    score_df_per_class = {}
    for cls in classes:

        in_mask = (y_train == cls) # Mask for in-class samples
        out_mask = ~in_mask # Mask for out-of-class samples

        X_ref = X_train[in_mask] 
        Y_ref = X_train[out_mask]

        s_fold = density_ratio_score(
                X_query=X_test,
                X_ref=X_ref,
                Y_ref=Y_ref,
                d=d,
                k=k,
                eps=eps,
            )

        df_cls = pd.DataFrame(
            s_fold,
            index=test_indices,
            columns=[cls]
        )
        df_cls.sort_index(inplace=True)
        score_df_per_class[cls] = df_cls
    score_df = pd.concat(score_df_per_class.values(), axis=1)
    score_df = score_df[classes]
    return score_df

def density_ratio_score(
    X_query,
    X_ref,
    Y_ref,
    d,
    k=4,
    eps=1e-7
):
    """
    Compute the density ratio score for each point in X_query based on the reference samples X_ref and Y_ref.

    The density ratio score (closely related to Renyi divergence) is a measure of how likely a point belongs to one distribution versus another.
    It is a non-parametric way of quantifying the separability of two distributions based on their samples. 
    The score is computed using the k-nearest neighbors (kNN) distances to the reference samples from each distribution.
  
    Parameters:
    ----------
    X_query : np.ndarray
        The query points for which we want to compute the density ratio scores, shape (Nq, d).
    X_ref : np.ndarray
        The reference samples from class A (in-class), shape (Nx, d).
    Y_ref : np.ndarray
        The reference samples from class B (out-of-class), shape (Ny, d).
    d : int
        The dimensionality of the data.
    k : int, optional
        The number of nearest neighbors to use in the density ratio score calculation. Default is 4.
        I've found that k doesn't have a huge effect on the scores, but it may depend on the dataset.
    eps : float, optional
        A small value to avoid division by zero in the density ratio score calculation. Default is 1e-7.
    
    Returns:
    -------
    s : np.ndarray
        The density ratio scores for each point in X_query, shape (Nq,).
    """

    N_x = X_ref.shape[0]
    N_y = Y_ref.shape[0]
    # kNN models (fit ONLY on reference data)
    nn_X = NearestNeighbors(n_neighbors=k).fit(X_ref)
    nn_Y = NearestNeighbors(n_neighbors=k).fit(Y_ref)

    # Distances to k-th neighbor
    X_k = nn_X.kneighbors(X_query)[0][:,k-1] + eps
    Y_k = nn_Y.kneighbors(X_query)[0][:,k-1] + eps

    # Pointwise density-ratio-like term
    s = np.log(N_y / N_x) + (d * np.log(Y_k / X_k))

    return s

