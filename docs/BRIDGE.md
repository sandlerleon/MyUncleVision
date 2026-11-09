# The MyUncle <-> MyUncleVision bridge

MyUncle ([github.com/sandlerleon/MyUncle](https://github.com/sandlerleon/MyUncle), doi:10.5281/zenodo.21223570) simulates a maintenance lattice: a rotor (angle theta) on every site, sites that are intact or damaged, repair and damage kinetics, an order-entropy bridge.
MyUncleVision treats any grid of embeddings (tissue patches, or anything else) as such a lattice. The two meet in `myuncle_vision/bridge.py`. MyUncle is used **unmodified** (vendored core, SHA-256 checked); the bridge only imports it.

| Direction | Function | What it does |
|---|---|---|
| MyUncle -> MyUncleVision | `bridge.simulate(preset, L, seed, steps, every)` | runs a MyUncle kinetic Monte Carlo (any of its presets) and returns snapshots `(step, theta, intact)` |
| | `bridge.order_maps(theta)` | MyUncleVision's local order maps for a MyUncle state: `Rbar`, `D`, `kappa`, `S_vmf`, `H_chi` (periodic, four neighbours; identical to MyUncle's `local_disorder` and `Bridge`) |
| | `bridge.analyse_myuncle_states(states)` | label-free novelty analysis of several MyUncle states with the MyUncleVision cohort pipeline (multi-scale appearance rarity + lattice smoothing): where is the organization unusual compared with the other states? |
| MyUncleVision -> MyUncle | `bridge.slide_to_myuncle(slide, Ls)` | an analysed slide (or any lattice of embeddings) becomes the initial state of a MyUncle MC (angle = first two principal components; damaged = locally disordered for its state); `mc.step()` then evolves it under MyUncle's kinetics. A **modelling analogy**, not prognosis |
| Check | `bridge.verify_bridge(mc)` | maximum absolute differences between the two implementations on a MyUncle state (entropy comparison skips sites where MyUncle's own I1/I0 overflows) |

Command line (writes `bridge_report.json` and `bridge_maps.png`: angle, D, H_chi and novelty per snapshot):

```
python -m myuncle_vision bridge --preset chiral_scaffold_biology --L 32 --steps 60 --every 15 --output out_bridge
```

Python:

```python
import myuncle_vision.bridge as B
snaps, mc = B.simulate("chiral_scaffold_biology", L=48, seed=0, steps=120, every=20)
print(B.verify_bridge(mc))                       # {'max_abs_diff_D': 3e-16, ..., 'equal': True}
res = B.analyse_myuncle_states(snaps)            # res[i]["novelty"], res[i]["order"]["D"], ...
```

## What the bridge is and is not

* **Equality, not analogy, where it is exact:** for d = 2 and MyUncle's neighbourhood, MyUncleVision's D, kappa, S_vmf and H_chi are MyUncle's (tests and `verify_bridge`); in d > 2 they are the von Mises-Fisher generalization.
* **Novelty is relative.** It is a percentile rank against the snapshots analysed together (a state is unusual compared with the others given), so give it several times, seeds or presets, and do not compare novelty values between runs. In a first-pass look at one preset the maps highlight domain-wall and defect regions, but that has not been validated against known defect locations.
* **Slide -> MyUncle is exploratory.** It maps a lattice of embeddings onto MyUncle's rotor lattice; the evolved states describe what MyUncle's kinetics would do to that arrangement, not what a tissue will do. It is off by default in the main pipeline (`--scenarios`).
* Use it for any future project that has a lattice of states and a MyUncle preset: simulate, analyse, compare presets, or feed real lattices of embeddings into MyUncle.
