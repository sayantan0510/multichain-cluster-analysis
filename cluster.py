#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import plotly.colors as pc
import plotly.graph_objects as go
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distance-based cluster analysis for multi-chain polymers."
    )
    parser.add_argument("-f", "--trajectory", required=True, help="GROMACS trajectory file (.xtc or .trr)")
    parser.add_argument("-s", "--tpr", required=True, help="GROMACS topology file (.tpr)")
    parser.add_argument("-c", "--cutoff-nm", required=True, type=float,
                        help="Distance cutoff in nm used to connect two chains")
    parser.add_argument("--atom-names", nargs="+", default=["BB", "CA"],
                        help="Atom names used for clustering. Default: BB CA")
    parser.add_argument("--min-pairs", type=int, default=1,
                        help="Minimum number of unique interchain atom pairs within cutoff required to connect two chains (default: 1)")
    parser.add_argument("--tmin-ps", type=float, default=0.0, help="Start time in ps (default: 0)")
    parser.add_argument("--tmax-ps", type=float, default=None, help="End time in ps (default: end of trajectory)")
    parser.add_argument("--out-prefix", default="multichain_", help="Output file prefix (default: multichain_)")
    parser.add_argument("--backend", choices=["serial", "OpenMP", "distopia"], default="serial",
                        help="Distance backend for MDAnalysis (default: serial)")
    parser.add_argument("--time-unit", choices=["ps", "ns", "us"], default="ps",
                        help="Time unit used in the HTML movie (default: ps)")
    parser.add_argument("--movie-plane", choices=["xy", "xz", "yz"], default="xy",
                        help="Projection plane for the HTML movie (default: xy)")
    parser.add_argument("--make-html-movie", action="store_true",
                        help="Also write an interactive HTML cluster movie")
    parser.add_argument("--movie-out", default=None,
                        help="Output HTML movie file (default: <out-prefix>cluster_movie.html)")
    parser.add_argument("--movie-stride", type=int, default=1,
                        help="Use every Nth analyzed frame in the HTML movie (default: 1)")
    parser.add_argument("--stride",type=int,default=1,
                        help="Analyze every Nth frame (default: 1)")    
    parser.add_argument("--verbose", action="store_true", help="Print progress information")
    return parser.parse_args()


def time_scale(unit: str) -> float:
    if unit == "ps":
        return 1.0
    if unit == "ns":
        return 1e-3
    if unit == "us":
        return 1e-6
    raise ValueError(f"Unknown time unit: {unit}")


def build_name_selection(atom_names: list[str]) -> str:
    cleaned = [name.strip() for name in atom_names if name.strip()]
    if not cleaned:
        raise ValueError("At least one valid atom name must be provided.")
    return " or ".join(f"name {name}" for name in cleaned)


def get_chain_groups(u: mda.Universe):
    atoms = u.atoms

    if hasattr(atoms, "molnums"):
        molnums = np.asarray(atoms.molnums)
        uniq = np.unique(molnums)
        if uniq.size > 1:
            return [atoms[molnums == m] for m in uniq]

    frags = atoms.fragments
    if len(frags) > 0:
        return [frag.atoms for frag in frags]

    raise RuntimeError("Could not identify chains from topology.")


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
    dr = r2 - r1
    lengths = box[:3]
    dr -= lengths * np.round(dr / lengths)
    return dr


def chain_masses_and_coms(chain_groups):
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


def unwrap_points(points: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points.copy()

    out = np.zeros_like(points)
    out[0] = points[0]

    for i in range(1, len(points)):
        out[i] = out[0] + minimum_image_vector(points[0], points[i], box_lengths)

    return out


def cluster_centers_from_chain_coms(components, coms, box_lengths):
    centers = np.zeros((len(components), 3), dtype=float)
    sizes = np.zeros(len(components), dtype=int)

    for cid, comp in enumerate(components):
        pts = coms[np.array(comp, dtype=int)]
        pts_unwrapped = unwrap_points(pts, box_lengths)
        centers[cid] = pts_unwrapped.mean(axis=0)
        sizes[cid] = len(comp)

    return centers, sizes


def recenter_points(points: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.copy()

    ref = points.mean(axis=0)
    shifted = points - ref
    shifted -= box_lengths * np.round(shifted / box_lengths)
    return shifted


def choose_plane(coords: np.ndarray, plane: str):
    if plane == "xy":
        return coords[:, 0], coords[:, 1], "x (A)", "y (A)"
    if plane == "xz":
        return coords[:, 0], coords[:, 2], "x (A)", "z (A)"
    return coords[:, 1], coords[:, 2], "y (A)", "z (A)"


def marker_sizes_from_population(populations: np.ndarray) -> np.ndarray:
    populations = populations.astype(float)
    return 22.0 + 18.0 * np.sqrt(populations)


def color_list(n: int):
    base = (
        pc.qualitative.Plotly
        + pc.qualitative.D3
        + pc.qualitative.G10
        + pc.qualitative.T10
        + pc.qualitative.Alphabet
    )
    return [base[i % len(base)] for i in range(n)]


def write_html_movie(movie_frames, plane: str, html_out: str):
    max_clusters = max(fd["nclusters"] for fd in movie_frames)
    palette = color_list(max_clusters)

    centered_all = np.concatenate([fd["centers"] for fd in movie_frames], axis=0)
    x_all, y_all, xlabel, ylabel = choose_plane(centered_all, plane)

    pad_x = 0.10 * (x_all.max() - x_all.min() + 1e-6)
    pad_y = 0.10 * (y_all.max() - y_all.min() + 1e-6)
    x_range = [float(x_all.min() - pad_x), float(x_all.max() + pad_x)]
    y_range = [float(y_all.min() - pad_y), float(y_all.max() + pad_y)]

    def make_trace(fd):
        x, y, _, _ = choose_plane(fd["centers"], plane)
        colors = [palette[i % len(palette)] for i in range(fd["nclusters"])]
        hover = [
            f"cluster = {cid}<br>population = {pop:d}<br>fraction = {frac:.1f}%"
            for cid, (pop, frac) in enumerate(zip(fd["sizes"], fd["fractions"]), start=1)
        ]

        return go.Scatter(
            x=x,
            y=y,
            mode="markers",
            text=hover,
            hoverinfo="text",
            marker=dict(
                size=marker_sizes_from_population(fd["sizes"]),
                color=colors,
                opacity=0.82,
                line=dict(width=1.0, color="black"),
            ),
        )

    plot_frames = []
    for i, fd in enumerate(movie_frames):
        title = (
            f"Cluster cartoon | time = {fd['time_label']} | "
            f"clusters = {fd['nclusters']} | largest = {fd['largest']}"
        )
        plot_frames.append(
            go.Frame(
                name=str(i),
                data=[make_trace(fd)],
                layout=go.Layout(title=title),
            )
        )

    initial = movie_frames[0]
    initial_title = (
        f"Cluster cartoon | time = {initial['time_label']} | "
        f"clusters = {initial['nclusters']} | largest = {initial['largest']}"
    )

    fig = go.Figure(
        data=[make_trace(initial)],
        layout=go.Layout(
            title=initial_title,
            xaxis=dict(title=xlabel, range=x_range, visible=False),
            yaxis=dict(title=ylabel, range=y_range, visible=False, scaleanchor="x", scaleratio=1),
            template="plotly_white",
            showlegend=False,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.02,
                    y=1.12,
                    direction="left",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "transition": {"duration": 0},
                                    "fromcurrent": True,
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    pad={"t": 35},
                    steps=[
                        dict(
                            method="animate",
                            label=f"{fd['time_value']:.2f}",
                            args=[
                                [str(i)],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        )
                        for i, fd in enumerate(movie_frames)
                    ],
                )
            ],
        ),
        frames=plot_frames,
    )

    fig.write_html(html_out, include_plotlyjs=True, full_html=True)


def main():
    args = parse_args()

    if args.cutoff_nm <= 0.0:
        raise ValueError("Cutoff must be positive.")
    if args.min_pairs < 1:
        raise ValueError("--min-pairs must be at least 1.")
    if args.movie_stride < 1:
        raise ValueError("--movie-stride must be at least 1.")

    cutoff_angstrom = args.cutoff_nm * 10.0
    selection_string = build_name_selection(args.atom_names)
    tscale = time_scale(args.time_unit)

    u = mda.Universe(args.tpr, args.trajectory)

    chain_groups = get_chain_groups(u)
    n_chains = len(chain_groups)

    if n_chains < 1:
        raise RuntimeError("No chains found in the system.")

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

    movie_frames = []
    movie_out = args.movie_out if args.movie_out is not None else f"{args.out_prefix}cluster_movie.html"

    cluster_dist_file = Path(f"{args.out_prefix}cluster_size_distribution.dat")

    traj = u.trajectory[::args.stride]

    iterator = tqdm(traj, desc="Analyzing frames", unit="frame") if args.verbose else traj

    with cluster_dist_file.open("w") as dist_fh:
        dist_fh.write("# time_ps cluster_size count\n")

        for ts in iterator:
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

            if args.make_html_movie and (len(times_ps) - 1) % args.movie_stride == 0:
                box_lengths = np.asarray(ts.dimensions[:3], dtype=float)
                centers, cluster_sizes = cluster_centers_from_chain_coms(components, coms, box_lengths)
                centers = recenter_points(centers, box_lengths)
                fractions = 100.0 * cluster_sizes / float(n_chains)
                display_time = time_ps * tscale

                movie_frames.append(
                    {
                        "time_value": display_time,
                        "time_label": f"{display_time:.3f} {args.time_unit}",
                        "centers": centers.copy(),
                        "sizes": cluster_sizes.copy(),
                        "fractions": fractions.copy(),
                        "nclusters": len(components),
                        "largest": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
                    }
                )

            if args.verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    clusters=nclusters,
                    largest=f"{largest_fraction:.1f}%",
                    rg=f"{largest_rg:.2f}nm",
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

    if args.make_html_movie:
        if not movie_frames:
            raise RuntimeError("HTML movie requested, but no frames were collected for the movie.")
        write_html_movie(movie_frames, args.movie_plane, movie_out)

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
        if args.make_html_movie:
            print(f"  {movie_out}")


if __name__ == "__main__":
    main()
