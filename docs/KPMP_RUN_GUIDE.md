# Running MyUncleVision on KPMP slides (the step that is still missing)

1. Register at https://www.herox.com/KidneyAI and get your personal download commands for the open KPMP Release 1 images (6,256 SVS files, about 2.4 TB). Use only open KPMP data.
2. Download a subset that fits your disk (a few dozen slides is enough for a first run). Put the `.svs` files in one directory.
3. Build and run (or use `pip install -r requirements.txt` and `python -m myuncle_vision run ...` without Docker):

   ```
   docker build -t myuncle-vision .
   docker run --rm -v D:\kpmp_subset:/input:ro -v D:\kpmp_results:/output myuncle-vision
   ```
   Useful flags (append after the image name): `run --input /input --output /output --max-patches 4000 --encoder dinov2 --hash-files`. Run it twice, once with `--encoder handcrafted`, and compare the findings.
   A GPU is strongly advised; on one CPU core a slide with 10,000 patches takes about ten minutes.
4. Look at `<output>/findings.csv`, `slides/<slide>/heatmaps.png` and `images_used.csv`. Check the axes separately (`--weights 1,0,0`, `0,1,0`, `0,0,1`) and a few values of `--rarity` (0.005, 0.01, 0.05).
5. Rebuild the documents with the real results: `python docs/build/build_documents.py --kpmp-run D:\kpmp_results`. This replaces the "KPMP RUN PENDING" notices in the HeroX pack.
6. Ask a renal pathologist or nephrologist to review the regions of the top findings, blinded to the score, and record for each whether it is known histology, an artifact, or neither. Without this the findings are not findings.
7. Complete the portal items that only you can do (cover-page rules statement, eligibility, team, repository link and commit ID dated on or before 29 April 2027).
