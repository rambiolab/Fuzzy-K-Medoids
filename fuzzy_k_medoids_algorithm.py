import numpy as np


# --- Fuzzy Clustering Algorithm ---

# K-means++ seeding adapted to k-medoids:
# Seed is index of sample that is first medoid
def _kmedoids_pp_init(d, k, rng):
    n = d.shape[0] #number of samples
    medoids = [rng.integers(n)] #assign the first medoid

    # choose next medoids with prob ∝ min distance^2 to existing medoids
    for _ in range(1, k): #start with 2, because the first medoid is already assigned
        mind = np.min(d[:, medoids], axis=1) #list of sample distances to its closest medoid
        probs = mind**2
        probs[medoids] = 0.0 #ensure medoids do not get repicked (probably redundant)
        s = probs.sum()
        if s == 0:
            # fall back: pick the farthest non-medoid
            cand = np.argmax(mind * (~np.isin(np.arange(n), medoids)))
        else:
            probs /= s #normalize probabilities
            cand = rng.choice(np.arange(n), p=probs) 
        medoids.append(int(cand))
    return np.array(medoids, dtype=int)


# PAM build algorithm for choosing medoids:
def _pam_build_init(d, k):
    n = d.shape[0]

    # Step 1: pick first medoid = global center (min total distance)
    total_distances = d.sum(axis=1)
    medoids = [np.argmin(total_distances)]

    # Step 2: greedily add remaining medoids
    while len(medoids) < k:
        best_candidate = None
        best_cost = np.inf

        for j in range(n):
            if j in medoids:
                continue
            candidate_medoids = medoids + [j]

            # cost = sum over points of distance to nearest medoid
            distances_to_medoids = np.min(d[:, candidate_medoids], axis=1)
            cost = distances_to_medoids.sum()

            if cost < best_cost:
                best_cost = cost
                best_candidate = j

        medoids.append(best_candidate)
    return np.array(medoids, dtype=int)


# Clustering algorithm
def fuzzy_k_medoids(d, n_clusters, m=2.0, init="build", max_iter=200, tol=1e-30, random_state=42, reinit_if_empty=False):
    """
    Fuzzy c-medoids (FCMdd) for a precomputed distance matrix D (n x n).
    Returns (U, medoids), where U is (n x k) fuzzy membership matrix.
    """
    assert d.shape[0] == d.shape[1], "d must be square."
    assert np.allclose(d, d.T), "d must be symmetric."
    n = d.shape[0]
    k = n_clusters
    rng = np.random.default_rng(random_state)
    eps = 1e-12

    if init == "build":
        medoids = _pam_build_init(d, k)
    elif init == "k-medoids++":
        medoids = _kmedoids_pp_init(d, k, rng)
    else:
        raise ValueError(f"init must be 'build' or 'k-medoids++', got {init!r}")
    U = np.full((n, k), 1.0 / k)

    for it in range(max_iter):
        U_old = U.copy()
        # distances to medoids
        dm = d[:, medoids].copy()
        dm[dm < eps] = eps  # protect zeros

        # membership update: Uik ∝ (1 / d(i, medoid_k))^(1/(m-1))
        p = 1.0 / (m - 1.0)
        invp = dm ** (-p)
        U = invp / invp.sum(axis=1, keepdims=True) #normalize memberships to 1

        # handle exact matches: if d(i, some medoid)=0, set hard 1 for that cluster
        zero_hits = (d[:, medoids] == 0.0)
        any_zero = zero_hits.any(axis=1)
        if any_zero.any():
            hard = np.argmax(zero_hits[any_zero], axis=1)
            U[any_zero] = 0.0
            U[any_zero, hard] = 1.0

        # (optional) guard against empty clusters by re-seeding their medoid
        if reinit_if_empty:
            wk = (U ** m).sum(axis=0)
            for kk in range(k):
                if wk[kk] < 1e-8:
                    # pick a new medoid far from current set
                    mind = np.min(d[:, medoids], axis=1)
                    cand = np.argmax(mind)
                    medoids[kk] = cand
                    dm = d[:, medoids].copy()
                    dm[dm < eps] = eps
                    invp = dm ** (-p)
                    U = invp / invp.sum(axis=1, keepdims=True)

        # medoid update: for each cluster, pick t minimizing sum_i u_ik^m * d(i, t)
        Um = U ** m
        new_medoids = medoids.copy()
        for kk in range(k):
            w = Um[:, kk]
            if w.sum() <= eps:
                continue
            # objective for candidate t: sum_i w_i * D[i, t]
            # restrict candidates to top members for speed (optional)
            # here: allow all n for simplicity
            obj = w @ d
            new_medoids[kk] = int(np.argmin(obj))
        medoids = new_medoids

        # convergence
        delta = np.linalg.norm(U - U_old)
        if delta < tol:
            #print("convergence after iteration", it)
            break

    return U, medoids


# --- Elbow plot fuzzy clustering ---
def fuzzy_objective(d, U, medoids, m=2.0):
    """
    Compute fuzzy clustering objective J_m.
    d : distance matrix (n x n)
    U : membership matrix (n x k)
    medoids : list of medoid indices
    """
    n, k = U.shape
    J = 0.0
    for kk in range(k):
        for i in range(n):
            J += (U[i, kk] ** m) * d[i, medoids[kk]]
    return J

def find_elbow(k_range, inertias):
    # Normalize to arrays
    ks = np.array(list(k_range))
    ys = np.array(inertias)

    # Line from first to last point
    p1 = np.array([ks[0], ys[0]])
    p2 = np.array([ks[-1], ys[-1]])

    # Distances from each point to the line
    distances = []
    for i in range(len(ks)):
        p = np.array([ks[i], ys[i]])
        # 2D cross product magnitude / line length
        # (written out, since np.cross no longer accepts 2D vectors in NumPy 2.x)
        v, w = p2 - p1, p1 - p
        d = np.abs(v[0] * w[1] - v[1] * w[0]) / np.linalg.norm(v)
        distances.append(d)

    elbow_idx = np.argmax(distances)
    return ks[elbow_idx], ys[elbow_idx], elbow_idx, p1, p2