# Tests

Regression guards for failures that produced **no error**. Each case here maps
to a bug that ran to completion and wrote plausible-looking output — the class
of failure that "does it crash?" cannot catch, and that this project has now
been bitten by three times.

```bash
pytest molplatte_preprocess/tests -q
```

This is deliberately not a coverage-driven suite. It pins the invariants whose
violation is silent: condvec label resolution, metadata key spellings across
sources, train/test split precedence, ligand validation floors, and
`HASH_VERSION`.
