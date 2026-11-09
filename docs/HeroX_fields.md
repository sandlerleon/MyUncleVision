# MyUncleVision: HeroX submission fields

Character counts are shown for every field (the portal allows about 3,000 characters per page, roughly 8.5 pages in all).

> **DRAFT, NOT READY TO SUBMIT.** [KPMP RUN PENDING] No KPMP whole-slide image has been analysed with this platform yet. This paragraph is replaced automatically by the results of the run when `python docs/build/build_documents.py --kpmp-run <output directory of the KPMP run>` is executed. Do not submit the pack with this notice in it.

## Before submitting

1. Register on HeroX (https://www.herox.com/KidneyAI) and obtain the personal download commands for the open KPMP images (Release 1: 6,256 SVS files, about 2.4 TB; start with a subset that fits your disk).
2. Run the container on the downloaded slides, then rebuild this pack with the run: python docs/build/build_documents.py --kpmp-run <output directory>.
3. Confirm eligibility (age 18 or older; U.S. citizen or permanent resident to receive a cash prize if you enter as an individual or team captain; not a federal employee acting in scope; no federal grant or contract funds used).
4. Copy the exact rules statement from the portal onto the cover page (the portal requires it; submissions without it are not evaluated). This pack does not make that agreement for you.
5. Push the repository, record the commit ID dated on or before the deadline (29 April 2027), and test the Docker image from a clean clone.
6. Look for a renal pathologist or nephrologist willing to review the top findings; criterion 3 asks whether domain experts are involved.
7. Check the IP terms: HeroX requires an irrevocable, nonexclusive licence to NIH; the repository is public under MIT.
8. Do not use HHS, NIH or NIDDK logos or claim federal endorsement.

## Public abstract (plain language; NIDDK may publish it) (1293 characters)

Kidney biopsies are photographed at very high resolution, but most of each image is never looked at closely, and computer programs are usually trained to find only what people have already named. MyUncleVision looks for the unnamed. It cuts each biopsy image into small tiles, describes every tile with a pretrained image model, and then measures how orderly the neighbourhood of every tile is, using a mathematical framework (MyUncle) developed for orderly and disorderly lattices. Places that look unusual, are unusually disordered for the kind of tissue they are, or make unusual contacts between kinds of tissue are ranked, grouped across biopsies, mapped on the slide and explained in plain numbers, so that a pathologist can check them. Artifacts such as blur, pen marks and folds are screened out first. The software runs from one command in a container and lists every image it used. It is tested on synthetic slides with planted answers, where, with a simple colour-and-texture description of the tiles, it ranks the planted structure well above normal tissue; with the pretrained image model it did not on that synthetic test, so its settings must be tuned and checked on real slides; it has not yet been run on kidney slides, and its candidates are for expert review, not diagnoses.

## Finding 1: Discovery and Novelty (50 points) (2994 characters)

What counts as a discovery here. A finding is a candidate organizational state of tissue: a group of regions, found in one or more slides, that is unusual in at least one of three measured ways and that survives four filters. The three measures are (1) appearance rarity: how far the tile's description is from the tiles of the rest of the cohort, judged at the scale of 1% of the cohort so that a rare state that recurs is still rare; (2) disorder for the tissue type: how much the tile disagrees with its neighbours, compared only with tiles of the same prototype tissue state, so that the ordinary borders between glomeruli, tubules and interstitium do not count; (3) rare contacts: how seldom the tile's tissue state touches the states around it. The four filters are artifact screening, a minimum region size, recurrence across slides (a state that appears in many biopsies is less likely to be one bad slide), and reproducibility when the reference set is split into two halves.

How a finding is separated from artifacts and routine histology. Artifact tiles (out of focus, pen, fold, glass) are removed before anything is ranked. Routine histology is handled by conditioning disorder on the tissue state and by the cohort-wide contact statistics. Each finding reports how many slides contain it and the entropy of that distribution, so a finding confined to one slide is labelled as such and a batch or scanner effect cannot be ruled out; the evaluation protocol compares recurrence with a null model in which the site or scanner is the explanatory variable.

Evidence available now. On synthetic slides with planted answers (not kidney tissue), with the handcrafted colour-and-texture encoder, the novelty score ranks planted anomalous tiles above normal tiles with an AUROC of 0.90 (slide-bootstrap interval 0.86–0.94), and 64% of the highest-scoring tiles lie within one tile of a planted anomaly against a base rate of 8.8%. These numbers say that the machinery works as designed; they do not say that the kidney contains anything novel.

With the DINOv2 encoder the same synthetic benchmark gives an AUROC of 0.22 for the novelty score, below chance: at the default rarity scale of 1% the planted states (about 3.1% of the patches) are not rare, and DINOv2 embeds them more homogeneously than normal tissue, so two of the three axes point the wrong way; the appearance axis alone recovers when the scale is raised (AUROC 0.91 at 10%). With DINOv2, 0 region(s) on slides with nothing planted entered a finding. The default settings are therefore not validated for DINOv2 on this benchmark, and a real run should compare encoders and inspect the axes separately.

Kidney results.
[KPMP RUN PENDING] No KPMP whole-slide image has been analysed with this platform yet. This paragraph is replaced automatically by the results of the run when `python docs/build/build_documents.py --kpmp-run <output directory of the KPMP run>` is executed. Do not submit the pack with this notice in it.

## Finding 2: Technical Approach and Performance (30 points) (3116 characters)

Approach. Each slide is read (OpenSlide for SVS), a tissue mask is made, and the slide is cut into non-overlapping tiles of 224 pixels at 0.5 micrometres per pixel; the tile grid is a lattice. Tiles failing quality control are excluded. Every tile is described by DINOv2 ViT-S/14 (a pretrained image model; no training on KPMP data). The descriptions are centred, reduced to 16 dimensions and normalized to a sphere. Local disorder is MyUncle's D = 1 minus the length of the mean of the neighbours' unit vectors; the entropy of the neighbourhood is the von Mises-Fisher entropy in d dimensions, which equals MyUncle's own entropy exactly at d = 2 (checked against the unmodified MyUncle code by automated tests). A weighted probabilistic OR of three rank-based axes gives the novelty score; regions are clustered across slides and ranked by novelty, recurrence and reproducibility. MyUncle's simulation is reused to run model scenarios of tissue progression from an observed slide (scenarios, not prognosis).

False-positive control. Artifact screening; state-wise disorder; the 1% rarity scale; minimum region size; recurrence; reproducibility under split reference sets; a spatial-shuffle null of the slide's overall order; and a refusal to report a finding as more than a candidate. On synthetic slides, quality control flagged 42% of the planted artifact tiles at a false-flag rate of 0.6% on normal tiles; on the two slides with nothing planted (one with only artifacts) the pipeline produced 5 and 5 candidate tiles (7.5 and 7.5 expected from the threshold by construction) and no finding.

Performance and reliability. The reference set for rarity excludes the tile's own slide neighbourhood, so no tile is compared with itself. On the synthetic benchmark (8 slides, 1,843 tiles; handcrafted encoder) the AUROC of the novelty score is 0.90 (interval 0.86–0.94); the appearance axis alone reaches 0.97 and disorder alone 0.84, so the combination is not better than its best axis there; it is kept because the axes see different things. These are development results: the thresholds and the rarity scale were adjusted after seeing synthetic data. Generalization to kidney tissue is untested.

With the DINOv2 encoder the same synthetic benchmark gives an AUROC of 0.22 for the novelty score, below chance: at the default rarity scale of 1% the planted states (about 3.1% of the patches) are not rare, and DINOv2 embeds them more homogeneously than normal tissue, so two of the three axes point the wrong way; the appearance axis alone recovers when the scale is raised (AUROC 0.91 at 10%). With DINOv2, 0 region(s) on slides with nothing planted entered a finding. The default settings are therefore not validated for DINOv2 on this benchmark, and a real run should compare encoders and inspect the axes separately.

Computing. The encoder dominates: about 15-30 tiles per second per CPU core, so a slide of ten thousand tiles takes about ten minutes on a CPU and minutes on a GPU; a patch budget per slide (--max-patches) and a resumable cache bound the work. The benchmark of 8 slides ran in 129 s on one CPU.

## Finding 3: Scientific Interpretability and Expert Usability (20 points) (1988 characters)

What an expert receives. For every slide: a four-panel heat-map figure (candidate regions, novelty, local disorder D, order information), a binary candidate mask, the novelty lattice, region boxes in the slide's own pixel coordinates (so they can be opened in any viewer), and a table with one row per tile (position, QC result, tissue state, the three axes, D and the entropy). For every finding: its type, size in square millimetres, the slides that contain it, its recurrence and reproducibility, and an explanation made only of measured numbers (for example: local disorder against the cohort median, hematoxylin and eosin density percentiles, the most frequent prototype state and its share of the cohort). No pathological term is assigned, because the method has no labels; every explanation ends with the request for expert review, and a finding that occurs in one slide only says so.

Connection to image features and patient stratification. Findings are defined on image features, and the per-slide summary (order against a spatial-shuffle null, disorder, damaged fraction, the model's rho-hat and whether it lies in MyUncle's bistable window) can be joined to patient-level data by the organizers; this submission makes no claim about patient groups.

Domain experts. No pathologist or nephrologist has been involved so far, and no output has been expert-reviewed. This is the main weakness of the submission under this criterion. The protocol asks for blinded review of the highest-ranked findings, with each assigned to known histology, artifact, or neither, and arranging a collaborator with kidney pathology expertise is a priority before submission (not yet arranged).

[KPMP RUN PENDING] No KPMP whole-slide image has been analysed with this platform yet. This paragraph is replaced automatically by the results of the run when `python docs/build/build_documents.py --kpmp-run <output directory of the KPMP run>` is executed. Do not submit the pack with this notice in it.

## Methods (about one page) (1555 characters)

Data selection: KPMP open (publicly released) whole-slide images only; controlled-access data are not used; the images actually used are listed with their file names, sizes and the way they were used in images_used.csv (written by every run). Preprocessing: tissue mask on a thumbnail; 224-pixel tiles at 0.5 micrometres per pixel (the grid is coarsened automatically when --max-patches is set); quality control (blur relative to the slide's median, pen ink, fold or debris, glass). Model design: DINOv2 ViT-S/14 (timm weights vit_small_patch14_dinov2.lvd142m) for tile embeddings; cohort PCA to 16 dimensions and k-means with 12 prototype states fitted at run time; the vendored, unmodified MyUncle core (third_party/myuncle/MyUncle_core.py, SHA-256 recorded) for the reference order-entropy definitions, the bistability analysis and the kinetic Monte Carlo of the progression scenarios. Training: none; everything is unsupervised and fitted on the input cohort at run time, so nothing is stored between runs and no label or clinical variable is used. Ranking: three cohort-rank axes (appearance rarity at the 1% scale, state-wise disorder, contact rarity), weighted probabilistic OR, regions of at least three tiles, clustering across slides, ranking by novelty, recurrence and reproducibility. Model files: the DINOv2 weights (embeddings; supports all findings), the MyUncle core (reference definitions and scenarios; supports the scenario analysis and the order/entropy checks), cohort PCA and k-means (cohort space and states; supports all findings).

## Validation (1276 characters)

Performance on unseen images: the pipeline has no training step, so every slide is unseen; on the synthetic benchmark the results are those above. On KPMP slides the study stage is not yet done and HeroX will run the code on unreleased slides. Uncertainty: slide-bootstrap interval of the AUROC (0.86–0.94); reproducibility of each finding under two disjoint reference halves; a spatial-shuffle null of each slide's order. Failure modes seen: DINOv2 at the default settings did not separate the planted structure on the synthetic benchmark (see the Approach narrative); a region of random texture is not disordered at the tile scale; a recurring novel state is invisible to nearest-neighbour distance at small k (fixed by judging rarity at 1% of the cohort); artifacts that overlap a tile only partly escape quality control; disorder is noisy with four neighbours and responds to compartment borders (hence the state-wise rank). Without quality control the share of artifacts among the top scores was 1.8% against 0.0% with it on the synthetic slides. Improvements: a pathology-pretrained encoder, stain normalization, larger neighbourhoods, a calibrated null, expert feedback. Computing resources: the benchmark ran on one CPU in 129 s; a GPU is advised for the full release.

## Team description (620 characters)

Leon Sandler, independent researcher (computational), sole author of the code and of MyUncle, the framework this work builds on. No pathologist, nephrologist or other kidney expert is part of the team at the time of writing; a collaboration for expert review of the findings has not yet been arranged and is a priority. All members are listed with name, affiliation and contact: Leon Sandler, Independent Researcher, Northbrook, Illinois, USA, sandler.leon@gmail.com. A generative AI assistant (Claude, Anthropic) helped write and test the code and draft the documents; the author reviewed them and takes responsibility.

## Visual evidence

- `docs/figures/fig_validation_example_handcrafted.png`: Synthetic slide (not kidney): planted anomalies (green), planted artifacts (red), the appearance and disorder axes and the novelty score on the tile lattice. Supports the performance narrative.
- `docs/figures/fig_pipeline.png`: Pipeline from slides to ranked, mapped and explained candidate states. Supports the technical narrative.
- `docs/figures/fig_bridge.png`: Order-information bridge in d dimensions and its agreement at d = 2 with the unmodified MyUncle core. Supports the technical narrative.

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

Total characters in the narrative fields: 12842 (about 4.3 pages at 3,000 characters per page).