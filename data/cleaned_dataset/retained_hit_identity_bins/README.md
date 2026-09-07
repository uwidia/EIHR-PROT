# Retained-hit identity bins

`assignments.csv` is the sole membership source for gate and predictive-performance comparisons. Both `sequence_homology_confidence_gate` and `sequence_homology_internal_gate` must filter their test predictions by `test_protein_id` and `category` from this same table; neither evaluation may recompute bins or change homology priors.

The categories are mutually exclusive and exhaustive over the authoritative test FASTA. `max_retained_identity` is the unrounded DIAMOND `pident` of the maximum-identity alignment among the final retained hits only. Missing identity is represented as an empty CSV cell exclusively for `no_retained_hit` rows.

`selected_alignment_query_coverage_fraction` is the 0--1 value used by the prior pipeline (and `selected_alignment_query_coverage_percent` is provided for inspection). `provenance.json` identifies the raw shared DIAMOND records and exact filtering/ranking policy. The original result did not request alignment-coordinate fields, so their audit columns in `assignments.csv` are intentionally empty rather than reconstructed from a new search.
