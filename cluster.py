#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distance-based cluster analysis for multi-chain polymers."
    )
    parser.add_argument(
        "-f", "--trajectory",
        required=True,
        help="GROMACS trajectory file (.xtc or .trr)"
    )
    parser.add_argument(
        "-s", "--tpr",
        required=True,
        help="GROMACS topology file (.tpr)"
    )
    parser.add_argument(
        "-c", "--cutoff-nm",
        required=True,
        type=float,
        help="Distance cutoff in nm used to connect two chains"
    )
    parser.add_argument(
        "--atom-names",
        nargs="+",
        default=["BB", "CA"],
        help=(
            "Atom names used for clustering. "
            "Default: BB CA. Example: --atom-names BB or --atom-names CA CB"
        )
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=1,
        help=(
            "Minimum number of unique interchain atom pairs within cutoff "
            "required to connect two chains (default: 1)"
        )
    )
    parser.add_argument(
        "--tmin-ps",
        type=float,
        default=0.0,
        help="Start time in ps (default: 0)"
    )
    parser.add_argument(
        "--tmax-ps",
        type=float,
        default=None,
        help="End time in ps (default: end of trajectory)"
    )
    parser.add_argument(
        "--out-prefix",
        default="multichain_",
        help="Output file prefix (default: multichain_)"
    )
    parser.add_argument(
        "--backend",
        choices=["serial", "OpenMP", "distopia"],
        default="serial",
        help="Distance backend for MDAnalysis (default: serial)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information"
    )
    return parser.parse_args()


def build_name_selection(atom_names: list[str]) -> str:
    if not atom_names:
        raise ValueError("At least one atom name must be provided.")

    cleaned = [name.strip() for name in atom_names if name.strip()]
    if not cleaned:
        raise ValueError("Atom names list is empty after cleaning.")

    return " or ".join(f"name {name}" for name in cleaned)


def get_chain_groups(u: mda.Universe):
    """
    Define chains from molnums if available, otherwise fall back to fragments.
    """
    atoms = u.atoms

    if hasattr(atoms, "molnums"):
        molnums = np.asarray(atoms.molnums)
        uniq = np.unique(molnums)
        if uniq.size > 1:
            return [atoms[molnums == m] for m in uniq]

    frags = atoms.fragments
    if len(frags) > 0:
        return [frag.atoms for frag in frags]

    raise RuntimeError(
        "Could not identify chains from topology. "
        "A .tpr with molecule information is strongly recommended."
    )


def connected_components(adjacency: list[list[int]]) -> list[list[int]]:
    n = len(adjacency)
    seen = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if seen[start]:
            continue

        queue = deque([start])
        seen[start] = True
        comp = []

        while queue:
            node = queue.popleft()
            comp.append(node)
            for nbr in adjacency[node]:
                if not seen[nbr]:
                    seen[nbr] = True
                    queue.append(nbr)

        components.append(comp)

    return components


def minimum_image_vector(r1: np.ndarray, r2: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Minimum-image displacement vector from r1 to r2.
    Assumes an orthorhombic box for the COM reconstruction step.
    """
    dr = r2 - r1
    lengths = box[:3]
    dr -= lengths * np.round(dr / lengths)
    return dr


def chain_masses_and_coms(chain_groups):
    """
    Return one mass and one COM per chain for the current frame.
    Uses all atoms in the chain for COM/Rg.
    """
    masses = np.zeros(len(chain_groups), dtype=float)
    coms = np.zeros((len(chain_groups), 3), dtype=float)

    for i, ag in enumerate(chain_groups):
        if hasattr(ag, "masses") and ag.masses is not None:
            m = np.asarray(ag.masses, dtype=float)
            total_mass = float(m.sum())
            if total_mass > 0.0:
                masses[i] = total_mass
                coms[i] = ag.center_of_mass()
                continue

        masses[i] = float(len(ag))
        coms[i] = ag.positions.mean(axis=0)

    return masses, coms


def unwrap_cluster_coms(cluster_indices, coms, box, adjacency):
    """
    Reconstruct cluster COM coordinates into one common image using BFS.
    """
    if len(cluster_indices) == 1:
        return np.array([coms[cluster_indices[0]]], dtype=float)

    cluster_set = set(cluster_indices)
    root = cluster_indices[0]

    placed = {root: coms[root].copy()}
    queue = deque([root])

    while queue:
        i = queue.popleft()
        for j in adjacency[i]:
            if j not in cluster_set or j in placed:
                continue
            placed[j] = placed[i] + minimum_image_vector(coms[i], coms[j], box)
            queue.append(j)

    return np.array([placed[idx] for idx in cluster_indices], dtype=float)


def radius_of_gyration(points: np.ndarray, weights: np.ndarray) -> float:
    wsum = weights.sum()
    if wsum <= 0.0:
        return 0.0

    center = np.sum(points * weights[:, None], axis=0) / wsum
    rg2 = np.sum(weights * np.sum((points - center) ** 2, axis=1)) / wsum
    return float(np.sqrt(max(rg2, 0.0)))


def build_adjacency(selected_chain_groups, box, cutoff_angstrom: float, min_pairs: int, backend: str):
    """
    Two chains are connected if at least `min_pairs` unique interchain atom pairs
    are within the cutoff.
    """
    n = len(selected_chain_groups)
    adjacency = [[] for _ in range(n)]

    for i in range(n - 1):
        pos_i = selected_chain_groups[i].positions
        if pos_i.shape[0] == 0:
            continue

        for j in range(i + 1, n):
            pos_j = selected_chain_groups[j].positions
            if pos_j.shape[0] == 0:
                continue

            dmat = distance_array(pos_i, pos_j, box=box, backend=backend)
            npairs = int(np.count_nonzero(dmat <= cutoff_angstrom))

            if npairs >= min_pairs:
                adjacency[i].append(j)
                adjacency[j].append(i)

    return adjacency


def write_two_column_dat(path: Path, header: str, x, y):
    with path.open("w") as fh:
        fh.write(header + "\n")
        for xi, yi in zip(x, y):
            fh.write(f"{xi:.6f} {yi:.10f}\n")


def plot_timeseries(x, y, xlabel, ylabel, title, outfile):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, lw=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()

    if args.cutoff_nm <= 0.0:
        raise ValueError("Cutoff must be positive.")
    if args.min_pairs < 1:
        raise ValueError("--min-pairs must be at least 1.")

    cutoff_angstrom = args.cutoff_nm * 10.0
    selection_string = build_name_selection(args.atom_names)

    u = mda.Universe(args.tpr, args.trajectory)

    chain_groups = get_chain_groups(u)
    n_chains = len(chain_groups)

    if n_chains < 1:
        raise RuntimeError("No chains found in the system.")

    # Use all atoms to define the chain, but only selected atom names for connectivity
    selected_chain_groups = [chain.select_atoms(selection_string) for chain in chain_groups]

    empty_chains = [i for i, ag in enumerate(selected_chain_groups) if len(ag) == 0]
    if empty_chains:
        raise RuntimeError(
            "Some chains contain no atoms matching the requested atom names. "
            f"Selection used: '{selection_string}'. "
            f"First problematic chain indices: {empty_chains[:10]}"
        )

    if args.verbose:
        print(f"Detected {n_chains} chains")
        print(f"Connectivity atom selection: {selection_string}")
        print(f"Minimum contacting pairs required: {args.min_pairs}")
        print(f"Cutoff: {args.cutoff_nm:.3f} nm")

    times_ps = []
    nclusters_series = []
    largest_rg_series = []
    largest_fraction_series = []
    mean_cluster_size_series = []

    cluster_dist_file = Path(f"{args.out_prefix}cluster_size_distribution.dat")
    with cluster_dist_file.open("w") as dist_fh:
        dist_fh.write("# time_ps cluster_size count\n")

        for ts in u.trajectory:
            time_ps = float(ts.time)

            if time_ps < args.tmin_ps:
                continue
            if args.tmax_ps is not None and time_ps > args.tmax_ps:
                break

            adjacency = build_adjacency(
                selected_chain_groups=selected_chain_groups,
                box=ts.dimensions,
                cutoff_angstrom=cutoff_angstrom,
                min_pairs=args.min_pairs,
                backend=args.backend,
            )

            components = connected_components(adjacency)
            sizes = np.array([len(comp) for comp in components], dtype=int)

            nclusters = int(len(components))
            mean_cluster_size = float(sizes.mean())

            largest_idx = int(np.argmax(sizes))
            largest_cluster = components[largest_idx]
            largest_fraction = 100.0 * len(largest_cluster) / n_chains

            masses, coms = chain_masses_and_coms(chain_groups)
            largest_cluster_coords = unwrap_cluster_coms(
                largest_cluster, coms, ts.dimensions, adjacency
            )
            largest_cluster_masses = masses[np.array(largest_cluster, dtype=int)]
            largest_rg = radius_of_gyration(largest_cluster_coords, largest_cluster_masses) / 10.0

            size_counts = Counter(sizes.tolist())
            for size in sorted(size_counts):
                dist_fh.write(f"{time_ps:.6f} {size:d} {size_counts[size]:d}\n")

            times_ps.append(time_ps)
            nclusters_series.append(nclusters)
            largest_rg_series.append(largest_rg)
            largest_fraction_series.append(largest_fraction)
            mean_cluster_size_series.append(mean_cluster_size)

            if args.verbose and len(times_ps) % 10 == 0:
                print(
                    f"time = {time_ps:10.3f} ps | "
                    f"clusters = {nclusters:4d} | "
                    f"largest Rg = {largest_rg:8.3f} nm | "
                    f"largest fraction = {largest_fraction:7.2f}%"
                )

    times_ps = np.asarray(times_ps, dtype=float)
    nclusters_series = np.asarray(nclusters_series, dtype=float)
    largest_rg_series = np.asarray(largest_rg_series, dtype=float)
    largest_fraction_series = np.asarray(largest_fraction_series, dtype=float)
    mean_cluster_size_series = np.asarray(mean_cluster_size_series, dtype=float)

    if times_ps.size == 0:
        raise RuntimeError("No frames were selected. Check --tmin-ps and --tmax-ps.")

    write_two_column_dat(
        Path(f"{args.out_prefix}nclusters.dat"),
        "# time_ps number_of_clusters",
        times_ps,
        nclusters_series,
    )
    write_two_column_dat(
        Path(f"{args.out_prefix}largest_cluster_rg.dat"),
        "# time_ps largest_cluster_rg_nm",
        times_ps,
        largest_rg_series,
    )
    write_two_column_dat(
        Path(f"{args.out_prefix}largest_cluster_fraction.dat"),
        "# time_ps largest_cluster_fraction_percent",
        times_ps,
        largest_fraction_series,
    )
    write_two_column_dat(
        Path(f"{args.out_prefix}mean_cluster_size.dat"),
        "# time_ps mean_cluster_size",
        times_ps,
        mean_cluster_size_series,
    )

    plot_timeseries(
        times_ps,
        nclusters_series,
        "Time (ps)",
        "Number of clusters",
        "Number of clusters vs time",
        f"{args.out_prefix}nclusters.png",
    )
    plot_timeseries(
        times_ps,
        largest_rg_series,
        "Time (ps)",
        "Largest cluster Rg (nm)",
        "Largest cluster size (Rg) vs time",
        f"{args.out_prefix}largest_cluster_rg.png",
    )
    plot_timeseries(
        times_ps,
        largest_fraction_series,
        "Time (ps)",
        "Chains in largest cluster (%)",
        "Largest cluster fraction vs time",
        f"{args.out_prefix}largest_cluster_fraction.png",
    )
    plot_timeseries(
        times_ps,
        mean_cluster_size_series,
        "Time (ps)",
        "Mean cluster size",
        "Mean cluster size vs time",
        f"{args.out_prefix}mean_cluster_size.png",
    )

    if args.verbose:
        print("Wrote:")
        print(f"  {args.out_prefix}nclusters.dat")
        print(f"  {args.out_prefix}nclusters.png")
        print(f"  {args.out_prefix}largest_cluster_rg.dat")
        print(f"  {args.out_prefix}largest_cluster_rg.png")
        print(f"  {args.out_prefix}largest_cluster_fraction.dat")
        print(f"  {args.out_prefix}largest_cluster_fraction.png")
        print(f"  {args.out_prefix}mean_cluster_size.dat")
        print(f"  {args.out_prefix}mean_cluster_size.png")
        print(f"  {args.out_prefix}cluster_size_distribution.dat")


if __name__ == "__main__":
    main()
