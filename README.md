# rf-detr-smol
RF-DETR architecture analysis for small objects detection.

## Research question
Where and why does RF-DETR fail on small / distant / visually weak objects?

## Hypothesis
Modern detection transformers may lose or under-prioritize small object features because of resolution bottlenecks, feature compression, query matching, and weak local detail representation.

## First goal
Run baseline inference and create a taxonomy of failure cases before modifying the model.
