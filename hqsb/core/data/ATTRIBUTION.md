# WikiText-2 Mini data attribution

Dataset text in `wikitext2-mini-v1.json` is separate from HQSB's Apache-2.0
source-code license. It is a small transformed selection of Wikipedia-derived
WikiText-2 test paragraphs curated by Stephen Merity, Caiming Xiong, James
Bradbury and Richard Socher (Salesforce Research).

Dataset: https://huggingface.co/datasets/Salesforce/wikitext
Paper: https://arxiv.org/abs/1609.07843

The pinned upstream dataset metadata declares CC BY-SA 3.0 and GFDL; its
README licensing paragraph separately links CC BY-SA 4.0. We preserve this
upstream discrepancy rather than relabeling the data as Apache-2.0. This copy
is attributed and shared under CC BY-SA 3.0 as declared in the dataset metadata:
https://creativecommons.org/licenses/by-sa/3.0/
Upstream README declaration: https://creativecommons.org/licenses/by-sa/4.0/
No endorsement by Wikipedia, Salesforce, or the original authors is implied.

Original article titles, original test-row numbers, source file hash and pinned
revision are recorded in the JSON. Changes are selecting eight paragraphs,
stripping surrounding whitespace, and wrapping them in reproducible JSON.
Scoring additionally uses each passage's first 64 model-specific tokens; no
translation or rewriting is performed. These are English regression samples,
not the full benchmark or a representative sample of all model abilities.

Rebuild with `scripts/console/build_wikitext2_mini.py` on an authorized execution
target using the source file and metadata whose identity is recorded in JSON.
