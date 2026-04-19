# Multi-chain Cluster Analysis

Distance-based cluster analysis for multi-chain polymer or protein systems using GROMACS(.xtc or .trr) trajectories. This code also produces a 
simple coarse-grained HTML movie to visualize the cluster size against time. 

This tool identifies clusters based on a user-defined distance cutoff the minimun number of unique pairs that defines a cluster. The GROMACS library, 'gmx clustsize' considers chain-i and chain-j to be in the same cluster, if any one atom/bead of chain-i falls within the distance cut-off of any bead of chain-j. This code allowes you to include them in the same cluster of 'n' number of sich unique pairs are present within the cut-off. Thus, we have a more robust way of defining clusters.

The idea is to give more control over what we call a 'cluster'. Often we see that bead-1 of chain-1 is within the cut-off from bead-100 of chain-2, with no other beads in the proximity. Should we still call chain-1 and chain-2 part of a cluster? Probably NOT. Here, if --min-pairs is set to say '3', chain-1 and chain-2 must have at least 3 unique pairs within the distance cut-off to be called a cluster.

---

## Features

- Distance-based clustering using minimum-image PBC
- Flexible atom selection (e.g., BB, CA, sidechains)
- Adjustable connectivity criterion (minimum number of contacting pairs, default is =1)
- Outputs data files(.dat), plots(.png), and movie(.html)

---

## Requirements

- Python ≥ 3.8  
- MDAnalysis  
- NumPy  
- Matplotlib
- plotly  

Install dependencies:

pip install MDAnalysis numpy matplotlib plotly

---

## Simple Usage

python cluster.py -f traj.xtc -s topol.tpr -c 0.6

---

## Command-Line Options

- -f : Trajectory file (.xtc or .trr)
- -s : Topology file (.tpr)
- -c : Distance cutoff in nm
- --atom-names : Atom names used for clustering (default: BB CA)
- --min-pairs : Minimum number of interchain atom pairs within cutoff (default: 1)
- --tmin-ps : Start time in ps (default: 0)
- --tmax-ps : End time in ps (default: full trajectory)
- --out-prefix : Output prefix (default: cluster)
- --verbose : Print progress
- --make-html-movie : Outputs an interactive .html movie
- --stride : Calculate clusters for every N-th frame
- --movie-stride : make movie frames every N-th frame
- --movie-plane : 2d projection plane (xy, yz, or zx)
- --time-unit : ps, ns, or us
- --movie-out : Output HTML movie file (default: <out-prefix>cluster_movie.html)

---

## Examples

Default:
python cluster.py -f traj.xtc -s topol.tpr -c 0.6

Stronger cluster connectivity:
python cluster.py -f traj.xtc -s topol.tpr -c 0.6 --min-pairs 3

Make interactive html movie:
python cluster.py -f traj.xtc -s topol.tpr -c 0.6 --min-pairs 3 --make-html-movie --movie-stride 10

---

## Output Files

- *_nclusters.dat
- *_nclusters.png
- *_largest_cluster_rg.dat
- *_largest_cluster_rg.png
- *_largest_cluster_fraction.dat
- *_largest_cluster_fraction.png
- *_mean_cluster_size.dat
- *_mean_cluster_size.png
- *_cluster_size_distribution.dat
- *_cluster_movie.html

---

## Viewing the outputs from terminal

- vim filename.dat
- eog filename.png
- xdg-open filename.html

---

## Interpretation

- Fewer clusters → more aggregation propensity
- Larger cluster fraction → phase separation
- Increasing Rg of the largest cluster → cluster growth (coalescence/ripening)
- multichain_cluster_size_distribution.dat file contains a detailed framewise information of the counts of different 'n-mers'. For example, a segment like this:

...
  
1000.0 1 4

1000.0 3 2

1000.0 15 1

...

means: at t=1000.0 ps there are 4 monomers, two trimers, and one 15-mer (largest cluster).

---

## Author

Dr. Sayantan Mondal  
homepage: https://sayantantheochem.weebly.com

---

## License

MIT License
