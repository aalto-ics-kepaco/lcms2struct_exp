# Script to run the different experiments

The scripts for into different categories:

## (1) Development

- Generally smaller (= faster) experiments for method development
- Possible usecases: 
  - trying out different hyper-parameter setups
  - determining default parameters for final experiments

## (2) Publication

- Run "publication ready" experimental setups, e.g. using the full dataset
- Generally are larger (= slower) experiments
- We want to run the experiments as few times as possible. 
- Results are used in the publication -> there should be an experiment for each section in the paper

## (3) Comparison

- "publication ready" experimental setups 
- runs for comparison methods: RT filtering, XLogP3 prediction, ...

## Debugging Setup

Each previously mentioned category also provides experiments / settings for debugging or "smoke test" purposes. Those
are typically used to ensure an experimental script is running correctly and producing the desired output. This has 
relevance for the experimentation on the cluster where resources are limited.
