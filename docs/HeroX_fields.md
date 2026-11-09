# MyUncleVision: HeroX submission fields

Character counts are shown for every field (the portal allows about 3,000 characters per page, roughly 8.5 pages in all).

> **DRAFT, NOT READY TO SUBMIT.** [KPMP RUN PENDING] No KPMP whole-slide image has been analysed with this platform yet. This paragraph is replaced automatically by the results of the run when `python docs/build/build_documents.py --kpmp-run <output directory of the KPMP run>` is executed. Do not submit the pack with this notice in it.

## Before submitting

1. Register on HeroX (https://www.herox.com/KidneyAI) and obtain the personal download commands for the open KPMP images (Release 1: 6,256 SVS files, about 2.4 TB; start with a subset that fits your disk).
2. Run the container on the downloaded slides (docs/KPMP_RUN_GUIDE.md), then rebuild this pack with the run: python docs/build/build_documents.py --kpmp-run <output directory>.
3. Confirm eligibility (age 18 or older; U.S. citizen or permanent resident to receive a cash prize if you enter as an individual or team captain; not a federal employee acting in scope; no federal grant or contract funds used).
4. Copy the exact rules statement from the portal onto the cover page (the portal requires it; submissions without it are not evaluated). This pack does not make that agreement for you.
5. Push the repository, record the commit ID dated on or before the deadline (29 April 2027), and test the Docker image from a clean clone.
6. Look for a renal pathologist or nephrologist willing to review the top findings blinded to the score; criterion 3 asks whether domain experts are involved.
7. Check the IP terms: HeroX requires an irrevocable, nonexclusive licence to NIH; the repository is public under MIT.
8. Do not use HHS, NIH or NIDDK logos or claim federal endorsement.

## Public abstract (plain language; NIDDK may publish it) (1092 characters)

Kidney biopsies are photographed at very high resolution, but most of each image is never looked at closely, and computer programs are usually trained to find only what people have already named. MyUncleVision looks for the unnamed. It cuts each biopsy image into small tiles, describes every tile with a pretrained image model, scores how rare each tile's appearance is, and smooths that score over the grid of tiles so that unusual regions, not isolated tiles, stand out. Regions are grouped across biopsies, mapped on the slide and explained in plain numbers, so that a pathologist can check them; artifacts such as blur, pen marks and folds are screened out first, and a mathematical framework (MyUncle) for orderly and disorderly lattices supplies measures of local disorder. The software runs from one command in a container and lists every image it used. It has been tested only on synthetic slides with planted answers, where it works well with one image model and poorly with another; it has not yet been run on kidney slides, and its candidates are for expert review, not diagnoses.

## Finding 1: Discovery and Novelty (50 points) (2683 characters)

What counts as a discovery here. A finding is a candidate organizational state of tissue: a group of regions, found in one or more slides, whose appearance is rare in the cohort and that survives four filters: artifact screening, a minimum region size, recurrence across slides (a state that appears in many biopsies is less likely to be one bad slide) and reproducibility when the reference set is split into two halves. Rarity is judged at several prevalence scales (0.5% to 10% of the cohort) because the frequency of an interesting state is not known in advance, and the score is smoothed over the tile grid so that compact regions are favoured.

How a finding is separated from artifacts and routine histology. Artifact tiles (out of focus, pen, fold, glass) are removed before anything is ranked. Each finding reports how many slides contain it and the entropy of that distribution, so a finding confined to one slide is labelled as such and a batch or scanner effect cannot be ruled out; the evaluation protocol compares recurrence with a null model in which the site or scanner is the explanatory variable. Two further measures from the lattice framework, local disorder for the tissue state and the rarity of contacts between tissue states, are reported for every tile as diagnostics: on the synthetic benchmark they helped with one image model and hurt with the other, so they do not drive the ranking.

Evidence available now (synthetic slides, not kidney tissue). Settings were chosen on a development cohort and frozen; a held-out cohort of 16 slides with three unseen anomaly types, about 0.8% prevalence and stain shifts was evaluated once. With the DINOv2 encoder the frozen score was the best of all methods tried (average precision 0.59, AUROC 0.99, region recall 1.00); with handcrafted features it was poor (average precision 0.04, region recall 0.29). On the development cohort a plain Mahalanobis baseline with the same smoothing scored higher than the frozen score (average precision 0.90 against 0.74 with DINOv2). The disorder and contact axes of the lattice helped with handcrafted features and hurt with DINOv2, so they are optional diagnostics and not part of the default score. These numbers say that the machinery can work as designed and that its choices are not yet robust; they do not say that the kidney contains anything novel.

Kidney results.
[KPMP RUN PENDING] No KPMP whole-slide image has been analysed with this platform yet. This paragraph is replaced automatically by the results of the run when `python docs/build/build_documents.py --kpmp-run <output directory of the KPMP run>` is executed. Do not submit the pack with this notice in it.

## Finding 2: Technical Approach and Performance (30 points) (2099 characters)

Approach. Each slide is read (OpenSlide for SVS), a tissue mask is made, and the slide is cut into non-overlapping tiles of 224 pixels at 0.5 micrometres per pixel; the tile grid is a lattice. Tiles failing quality control are excluded. Every tile is described by DINOv2 ViT-S/14 (a pretrained image model; no training on KPMP data), reduced to 16 dimensions and normalized to a sphere. Appearance rarity is the mean distance to the nearest tiles of a reference sample of the cohort, at five scales, taking the highest rank; the score is then averaged over each tile's 3 x 3 neighbourhood. Local disorder is MyUncle's D = 1 minus the length of the mean of the neighbours' unit vectors; the von Mises-Fisher entropy in d dimensions equals MyUncle's own entropy exactly at d = 2 (checked against the unmodified MyUncle code by automated tests). Regions are clustered across slides and ranked by novelty, recurrence and reproducibility.

Comparison and false-positive control. On the same embeddings the score was compared with Isolation Forest, Local Outlier Factor, nearest-neighbour distance and Mahalanobis distance, each raw and smoothed in the same way. Smoothing raised the spatial coherence of every scorer. On the development cohort the frozen score beat the nonparametric baselines but not smoothed Mahalanobis (average precision 0.90 and 0.74 with DINOv2). On the held-out cohort (settings frozen beforehand, evaluated once) the frozen score was the best method with DINOv2 (average precision 0.59, 0.38 false regions per slide) and a poor one with handcrafted features (0.04); with handcrafted features it was also unstable to a coarser patch size. Seeds, embedding size and number of states changed the flagged set little.

Computing. The encoder dominates: about 15 to 30 tiles per second per CPU core, so a slide of ten thousand tiles takes about ten minutes on a CPU and minutes on a GPU; a patch budget per slide (--max-patches) and a resumable cache bound the work. The Docker image was built and run end to end on three slides with its default command, offline, in about two minutes.

## Finding 3: Scientific Interpretability and Expert Usability (20 points) (1951 characters)

What an expert receives. For every slide: a four-panel heat-map figure (candidate regions, novelty, local disorder D, order information), a binary candidate mask, the novelty lattice, region boxes in the slide's own pixel coordinates (so they can be opened in any viewer), and a table with one row per tile (position, QC result, tissue state, the axes, D and the entropy). For every finding: its type, size in square millimetres, the slides that contain it, its recurrence and reproducibility, and an explanation made only of measured numbers (local disorder against the cohort median, hematoxylin and eosin density percentiles, the most frequent prototype state and its share of the cohort). No pathological term is assigned, because the method has no labels; every explanation ends with the request for expert review, and a finding that occurs in one slide only says so.

Connection to image features and patient stratification. Findings are defined on image features, and the per-slide summary (order against a spatial-shuffle null, disorder, candidate counts) can be joined to patient-level data by the organizers; this submission makes no claim about patient groups.

Domain experts. No pathologist or nephrologist has been involved so far, and no output has been expert-reviewed. This is the main weakness of the submission under this criterion. The protocol asks for blinded review of the highest-ranked findings by at least two experts, with each assigned to known histology, artifact, or neither, and with inter-rater agreement; arranging a collaborator with kidney pathology expertise is a priority before submission (not yet arranged).

[KPMP RUN PENDING] No KPMP whole-slide image has been analysed with this platform yet. This paragraph is replaced automatically by the results of the run when `python docs/build/build_documents.py --kpmp-run <output directory of the KPMP run>` is executed. Do not submit the pack with this notice in it.

## Methods (about one page) (1457 characters)

Data selection: KPMP open (publicly released) whole-slide images only; controlled-access data are not used; the images actually used are listed with their file names, sizes and the way they were used in images_used.csv (written by every run). Preprocessing: tissue mask on a thumbnail; 224-pixel tiles at 0.5 micrometres per pixel (the grid is coarsened automatically when --max-patches is set); quality control (blur relative to the slide's median, pen ink, fold or debris, glass). Model design: DINOv2 ViT-S/14 (timm weights vit_small_patch14_dinov2.lvd142m) for tile embeddings; cohort PCA to 16 dimensions and k-means with 12 prototype states fitted at run time; the vendored, unmodified MyUncle core (third_party/myuncle/MyUncle_core.py, SHA-256 recorded) for the reference order-entropy definitions. Training: none; everything is unsupervised and fitted on the input cohort at run time, so nothing is stored between runs and no label or clinical variable is used. Ranking: multi-scale appearance rarity (nearest-neighbour distance to a cohort reference sample), 3 x 3 lattice smoothing, regions of at least three tiles, clustering across slides, ranking by novelty, recurrence and reproducibility. Model files: the DINOv2 weights (embeddings; supports all findings), cohort PCA, k-means and the reference sample (cohort space, states, rarity; support all findings), the MyUncle core (reference definitions; supports the order and entropy diagnostics).

## Validation (1199 characters)

Performance on unseen images: the pipeline has no training step, so every slide is unseen; settings were frozen on a development cohort and evaluated once on a held-out synthetic cohort (results above and in the repository). HeroX will run the code on unreleased KPMP slides. Uncertainty: slide-bootstrap intervals of the AUROC, reproducibility of each finding under two disjoint reference halves, robustness to seeds, embedding size, states, weights, smoothing and patch size. Failure modes seen: with the pretrained model the original three-axis score at a single rarity scale scored below chance because the planted states were more common than the scale and the encoder embedded them tightly (fixed by multi-scale rarity, found on the development cohort); with handcrafted features the frozen score diluted small anomalies on the held-out cohort and collapsed at a coarser patch size; partly blurred tiles escape quality control; a single-Gaussian baseline was competitive. Improvements: a pathology-pretrained encoder, stain normalization, smoothing of the raw score, a calibrated null, expert feedback. Computing resources: the benchmarks ran on one CPU; a GPU is advised for the full release.

## Team description (620 characters)

Leon Sandler, independent researcher (computational), sole author of the code and of MyUncle, the framework this work builds on. No pathologist, nephrologist or other kidney expert is part of the team at the time of writing; a collaboration for expert review of the findings has not yet been arranged and is a priority. All members are listed with name, affiliation and contact: Leon Sandler, Independent Researcher, Northbrook, Illinois, USA, sandler.leon@gmail.com. A generative AI assistant (Claude, Anthropic) helped write and test the code and draft the documents; the author reviewed them and takes responsibility.

## Visual evidence

- `docs/figures/fig_heldout_example.png`: Held-out synthetic slide (not kidney): planted anomalies (green), planted artifact (red), unsmoothed appearance score, the frozen score and a smoothed Mahalanobis baseline. Supports the performance narrative.
- `docs/figures/fig_heldout.png`: AUROC and average precision of the main methods on the held-out synthetic cohort, both encoders. Supports the performance narrative.
- `docs/figures/fig_pipeline.png`: Pipeline from slides to ranked, mapped and explained candidate states. Supports the technical narrative.

## CSV of KPMP images used

`images_used.csv`, written by every run into the output directory.

## Citations (1759 characters)

[1] M. Oquab, T. Darcet, T. Moutakanni, et al., DINOv2: learning robust visual features without supervision, Transactions on Machine Learning Research (2024), arXiv:2304.07193. https://doi.org/10.48550/arXiv.2304.07193
[2] A. Goode, B. Gilbert, J. Harkes, D. Jukic, M. Satyanarayanan, OpenSlide: a vendor-neutral software foundation for digital pathology, Journal of Pathology Informatics 4 (2013) 27. https://doi.org/10.4103/2153-3539.119005
[3] B.B. Lake, R. Menon, S. Winfree, et al., An atlas of healthy and injured cell states and niches in the human kidney, Nature 619 (2023) 585–594. https://doi.org/10.1038/s41586-023-05769-3
[4] A. Banerjee, I.S. Dhillon, J. Ghosh, S. Sra, Clustering on the unit hypersphere using von Mises-Fisher distributions, Journal of Machine Learning Research 6 (2005) 1345–1382.
[5] K.V. Mardia, P.E. Jupp, Directional Statistics, Wiley, Chichester (2000).
[6] F. Pedregosa, G. Varoquaux, A. Gramfort, et al., Scikit-learn: machine learning in Python, Journal of Machine Learning Research 12 (2011) 2825–2830.
[7] M. Lochner, B.A. Bassett, Astronomaly: personalised active anomaly detection in astronomical data, Astronomy and Computing 36 (2021) 100481. https://doi.org/10.1016/j.ascom.2021.100481
[8] D. Baron, D. Poznanski, The weirdest SDSS galaxies: results from an outlier detection algorithm, Monthly Notices of the Royal Astronomical Society 465 (2017) 4530–4555. https://doi.org/10.1093/mnras/stw3021
[9] L. Sandler, MyUncle: a compact maintenance-lattice framework, software, Zenodo (2026). https://doi.org/10.5281/zenodo.21223570
[10] A.C. Ruifrok, D.A. Johnston, Quantification of histochemical staining by color deconvolution, Analytical and Quantitative Cytology and Histology 23 (2001) 291–299 (PMID 11531144).

Total characters in the narrative fields: 11101 (about 3.7 pages at 3,000 characters per page).