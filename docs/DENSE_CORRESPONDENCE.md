# Section 2 dense-correspondence protocol

The dense experiment uses the canonical image-disjoint NAVI-Wild
correspondence split in three relative-viewpoint bins: 0--40, 40--80, and
80--120 degrees.

For every complete object/depth-valid source patch, the evaluator searches the
full unmasked target patch grid. It stores directional top-K candidates,
mutual-entry ranks, mutual graph sizes, and float64 3D errors. The main
correspondence-recall value is the equal mean of strict 1, 2, and 5 cm recalls
after hierarchical direction→pair→object→bin aggregation.

The released sweep is:

- DINOv3 ViT-L/16 layers 16--24 and positional-debias ranks
  0, 100, 200, 300, 400, 500, 600;
- raw DINOv2 ViT-L/14-register layers 16--24 at rank 0;
- progressive mutual K values 1--8;
- no association-count cap.

Figure 2 uses corrected DINOv3 rank 200 and raw DINOv2 layer 24. The plotter
also emits the complete layer/rank reports needed to inspect that selection.

Run the commands in [`experiments/README.md`](../experiments/README.md). Each
bin is resumable, but an output root must contain exactly the requested
pair/layer rectangle; unexpected shards fail closed to prevent mixing runs.

Expected generated layout:

```text
SECTION2_ROOT/
├── bin_0-40/
│   ├── split_audit.json
│   ├── shards/layer16/pair_*.npz
│   └── ...
├── bin_40-80/
└── bin_80-120/
```

Raw descriptors, positional bases, and shards can be large and are ignored by
Git. Compact reports and figures should be regenerated from a local artifact
tree rather than committed.
