# Fuzzy k-medoids clustering

A small NumPy implementation of **fuzzy k-medoids** (fuzzy c-medoids, FCMdd) for a **precomputed distance matrix**. Every sample gets a membership degree in each cluster instead of a single hard label, and each cluster is represented by a real sample (its medoid) rather than an average.

Because it only needs pairwise distances, it works with any dissimilarity: for example `1 − Jaccard` or `1 − ANI` from sketching tools such as Mash or sourmash, Bray–Curtis distances, or Euclidean distances. It was written to group metagenomic samples by sketch similarity before co-assembly, but nothing in the code is specific to that use.

This is our own implementation of **fuzzy c-medoids** (FCMdd; Krishnapuram et al. 2001), the usual name of the method in the literature. The "c" is the traditional letter for the number of clusters in fuzzy clustering, as in fuzzy c-means; here the method is called fuzzy k-medoids, after the hard k-medoids method it generalises.

The objective, the membership and medoid updates, and the stopping rule follow FCMdd. Our implementation differs in three ways:

- **Initialisation:** PAM BUILD (default) or k-medoids++ seeding. The paper uses random medoids, or the most central sample followed by farthest-first selection.
- **Zero distances:** a sample at distance 0 from a medoid gets membership 1 in that cluster, which the paper's membership formula leaves undefined.
- **Full medoid search:** every sample is a medoid candidate. The paper's faster linearised variant (LFCMdd), which only considers each cluster's highest-membership samples, and its outlier-trimming robust variant (RFCMdd) are not included.

The module also contains helpers to choose the number of clusters with an elbow plot.

## Contents

| function | purpose |
|---|---|
| `fuzzy_k_medoids` | the clustering algorithm |
| `fuzzy_objective` | the objective J<sub>m</sub> of a fitted clustering (for elbow plots) |
| `find_elbow` | picks the elbow of an objective-vs-k curve |
| `_pam_build_init`, `_kmedoids_pp_init` | medoid initialisation (internal) |

## Requirements

- Python 3
- NumPy

There is nothing to install: copy `fuzzy_k_medoids_algorithm.py` into your project and import it.

## How it works

Given an *n × n* distance matrix *d*, *k* clusters and a fuzzifier *m* > 1:

1. **Initialisation.** Choose *k* starting medoids, either
   - `"build"` (default): the PAM BUILD step. The first medoid is the sample with the smallest total distance to all others; each next medoid is the sample that most reduces the summed distance of all samples to their nearest medoid. Deterministic.
   - `"k-medoids++"`: k-means++ seeding on the distance matrix. The first medoid is random; each next one is drawn with probability proportional to the squared distance to its nearest existing medoid. Uses `random_state`.
2. **Membership update.** Each sample's membership in cluster *j* is

   u<sub>ij</sub> = (1 / d(i, medoid<sub>j</sub>))<sup>1/(m−1)</sup> / Σ<sub>l</sub> (1 / d(i, medoid<sub>l</sub>))<sup>1/(m−1)</sup>

   so memberships of a sample sum to 1. A sample at distance 0 from a medoid (the medoid itself, or an exact duplicate) gets membership 1 in that cluster and 0 elsewhere.
3. **Medoid update.** For each cluster, the new medoid is the sample *t* that minimises Σ<sub>i</sub> u<sub>ij</sub><sup>m</sup> · d(i, t), searched over all samples.
4. Repeat 2–3 until the memberships stop changing (or `max_iter` is reached).

The objective being minimised is

J<sub>m</sub> = Σ<sub>j</sub> Σ<sub>i</sub> u<sub>ij</sub><sup>m</sup> · d(i, medoid<sub>j</sub>)

which `fuzzy_objective` returns.

## Usage

### Basic clustering

```python
import numpy as np
from fuzzy_k_medoids_algorithm import fuzzy_k_medoids

# Example: a similarity matrix from `sourmash compare -o sim.npy`
sim = np.load("sim.npy")
d = 1.0 - sim                     # similarity -> distance
d = (d + d.T) / 2                 # make sure it is exactly symmetric
np.fill_diagonal(d, 0.0)

U, medoids = fuzzy_k_medoids(d, n_clusters=10, m=2.0, init="build")

U.shape        # (n_samples, 10): membership of each sample in each cluster
medoids        # indices of the 10 medoid samples
```

### Turning memberships into clusters

The function returns fuzzy memberships only. Two common ways to use them:

```python
# Hard assignment: each sample to its highest-membership cluster
labels = U.argmax(axis=1)

# Overlapping clusters: a sample joins every cluster where its membership
# exceeds a threshold (choose the threshold for your application)
threshold = 0.3
members = [np.where(U[:, j] >= threshold)[0] for j in range(U.shape[1])]
```

`U.max(axis=1)` shows how confidently each sample belongs to its best cluster; values close to 1/k mean the sample sits between clusters.

### Choosing the number of clusters (elbow method)

```python
from fuzzy_k_medoids_algorithm import fuzzy_k_medoids, fuzzy_objective, find_elbow

k_range = range(2, 41)
objectives = []
for k in k_range:
    U, medoids = fuzzy_k_medoids(d, n_clusters=k)
    objectives.append(fuzzy_objective(d, U, medoids))

best_k, best_obj, idx, p1, p2 = find_elbow(k_range, objectives)
```

`find_elbow` draws a straight line from the first to the last point of the curve and returns the point farthest from it. It returns the elbow's k, its objective value, its index in `k_range`, and the two end points of the line (`p1`, `p2`) for plotting:

```python
import matplotlib.pyplot as plt

plt.plot(list(k_range), objectives, "o-")
plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color="grey")
plt.axvline(best_k, color="red")
plt.xlabel("number of clusters k")
plt.ylabel("objective $J_m$")
plt.show()
```

## API

### `fuzzy_k_medoids(d, n_clusters, m=2.0, init="build", max_iter=200, tol=1e-30, random_state=42, reinit_if_empty=False)`

| parameter | default | description |
|---|---|---|
| `d` | | square, symmetric NumPy array of pairwise distances (checked with `assert`) |
| `n_clusters` | | number of clusters *k* |
| `m` | `2.0` | fuzzifier, must be > 1; controls how soft the memberships are |
| `init` | `"build"` | starting medoids: `"build"` (deterministic) or `"k-medoids++"` (random) |
| `max_iter` | `200` | maximum number of update rounds |
| `tol` | `1e-30` | convergence threshold on the change in `U` |
| `random_state` | `42` | seed for `"k-medoids++"`; ignored for `"build"` |
| `reinit_if_empty` | `False` | re-seed the medoid of a cluster that has lost (almost) all weight |

**Returns** `(U, medoids)`: `U` is an *n × k* membership matrix whose rows sum to 1; `medoids` is an array of *k* sample indices.

#### The fuzzifier `m`

`m` sets how strongly a sample's membership is concentrated on its nearest medoid. It enters the algorithm in two places:

- **Memberships:** u<sub>ij</sub> ∝ d(i, medoid<sub>j</sub>)<sup>−1/(m−1)</sup>. With the default m = 2, membership is simply proportional to 1/distance. With m = 1.5 it is proportional to 1/distance², so the nearest medoid dominates; with m = 3 it is proportional to 1/√distance, so memberships are spread more evenly.
- **Medoid update and objective:** each sample is weighted by u<sub>ij</sub><sup>m</sup>. Since memberships are below 1, a larger m shrinks the weight of samples that are only partly in a cluster, so the medoid is chosen mainly by the cluster's clear members.

Limits:

- **m → 1:** memberships become 0 or 1, i.e. hard k-medoids.
- **m → ∞:** every membership tends to 1/k, and the clustering carries no information.

The default m = 2 is the usual choice for fuzzy c-means. For FCMdd, Krishnapuram et al. (2001) recommend m between 1 and 1.5, because at higher values the medoids tend to stop moving between iterations. Memberships also depend on the scale of the distances, so choose m for your own data, for example by checking how many samples get a clear highest membership. Use the same m in `fuzzy_objective` as in the fit, since the objective depends on it.

#### Initialisation: `init` and `random_state`

- `"build"` gives the same result every time and is a good default. Its cost grows with k and with the square of n (see *Notes and limitations*).
- `"k-medoids++"` is faster to initialise but random. The result can depend on the seed, so run several `random_state` values and keep the run with the lowest `fuzzy_objective`.

#### Stopping: `max_iter` and `tol`

Each iteration updates the memberships from the current medoids and then updates the medoids from those memberships. Because medoids are actual samples, they change in discrete steps, and the memberships are fully determined by the medoids. So the change in `U` between iterations is either exactly 0 (the medoids did not change) or clearly larger than 0.

- **`tol`:** the run stops when the Frobenius norm of the change in `U` is below `tol`. With the default 1e-30, this means: stop as soon as the medoids no longer change. Any very small value behaves the same way; a larger value (e.g. 1e-3) may stop while the medoids are still moving and is not recommended.
- **`max_iter`:** an upper limit on the number of iterations. Convergence usually takes only a few; Krishnapuram et al. (2001) report about 5–6 iterations. The limit mainly guards against medoids alternating between two configurations.
- The function does **not warn** when it stops at `max_iter` without converging, and it does not return the number of iterations used. In that case the returned `U` was computed from the medoids of the previous step, so `U` and `medoids` may not match exactly. If you suspect this, rerun with a larger `max_iter` and check that the medoids are unchanged.

#### `reinit_if_empty`

If `True`, a cluster whose total weight Σ<sub>i</sub> u<sub>ij</sub><sup>m</sup> falls below 1e-8 gets a new medoid: the sample farthest from all current medoids. With the inverse-distance memberships used here, every sample keeps some membership in every cluster, so this rarely triggers; it is off by default.

### `fuzzy_objective(d, U, medoids, m=2.0)`

Returns J<sub>m</sub> for a fitted clustering. Use the same `m` as in the fit.

### `find_elbow(k_range, inertias)`

Returns `(k, value, index, p1, p2)` for the point of `inertias` farthest from the line between its first and last points.

## Notes and limitations

- **Distances, not similarities.** Convert similarities first (e.g. `1 − similarity`). Distances should be non-negative with zeros on the diagonal.
- **Memory and run time** grow with the square of the number of samples, since the full distance matrix is held in memory and each medoid update scans all samples. The BUILD initialisation is the slowest part for large *k*. This is fine for hundreds to a few thousand samples.
- **Every sample has a non-zero membership in every cluster**, except samples at distance 0 from a medoid. A cluster therefore never becomes completely empty, so `reinit_if_empty` rarely triggers.
- **Two clusters can end up with the same medoid**; the algorithm does not prevent this. Check `len(set(medoids)) == k` if it matters.
- **Local optimum.** Like k-means, the result depends on the starting medoids. `"build"` is deterministic; with `"k-medoids++"` try several `random_state` values and keep the run with the lowest objective.
- **The objective decreases with k**, so it cannot be compared across k directly; use the elbow or an external criterion.

## References

- Krishnapuram R, Joshi A, Nasraoui O, Yi L (2001). Low-complexity fuzzy relational clustering algorithms for Web mining. *IEEE Transactions on Fuzzy Systems* 9(4):595–607. (fuzzy c-medoids)
- Kaufman L, Rousseeuw PJ (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley. (PAM and its BUILD step)
- Arthur D, Vassilvitskii S (2007). k-means++: the advantages of careful seeding. *Proceedings of the 18th ACM-SIAM Symposium on Discrete Algorithms*, 1027–1035.


