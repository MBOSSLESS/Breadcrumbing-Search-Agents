# SafeSearch data

This directory distributes case indices only. Task records are retrieved from the upstream [SafeSearch dataset](https://huggingface.co/datasets/jianshuod/safesearch). `source.json` pins an immutable upstream revision and SHA-256 checksums for five files. Refer to the upstream dataset for its terms and attribution.

From the repository root, using Python 3.10 or newer:

```bash
python scripts/prepare_data.py
```

This downloads only the five pinned source files, verifies their checksums, and writes family JSONL files under `data/safesearch/{train,test,runtime_test20,tgse_test20}/`. It requires only the Python standard library. Existing files with the same output names are overwritten. Generated records retain all upstream fields without alteration.

For offline preparation, pass a local Hugging Face dataset checkout at the pinned revision, or a SafeSearch repository checkout containing `assets/test-cases/filtered-by-gpt-4.1-mini/`:

```bash
python scripts/prepare_data.py --source /path/to/upstream-checkout
```

Both source layouts have exact byte checksums in `source.json`. `--output /path/to/output` changes the generated data directory. A missing file or checksum mismatch stops preparation before output files are written.

The ordered indices in `splits/` contain only source family and case ID; array position defines order within each family. There are 100 training cases (20 per family), 187 test cases, and 13 excluded cases, covering all 300 upstream cases. The test family counts are 40 Advertisement, 40 Bias Inducing, 40 Fake Information, 29 Harmful Output, and 38 Prompt Injection. Exclusions remove queries that explicitly constrain retrieval to a named site, domain, official website, or platform page. The fixed indices define the split; preparation does not sample again.

Preparation also writes two recorded evaluation subsets, each with 20 held-out cases per family (100 total). `runtime_test20/` is used by runtime ablations and the additional baseline comparisons; it preserves the first 20 cases in each ordered test split. `tgse_test20/` contains the recorded TGSE evaluation cohort, shared by the four TGSE regime comparisons. These cohorts differ. Their exact case IDs and order are distributed in `splits/runtime_test20.json` and `splits/tgse_test20.json`. The Bias Inducing adv-warm comparison uses the same aligned 20 cases as the other TGSE regimes, selected from its larger 40-case evaluation. Neither subset changes the full training or test split.
