# TabM and Transformer specialist comparison

This one-seed experiment compares two architecture families on the frozen
training and validation contracts:

- shared five-value forward prediction plus five direction outputs;
- numerical inverse ranking over the fixed 81-action candidate set.

The completed training sources contain 10,500 setup groups and 850,500
transitions. Old-IID and difficult validation groups remain frozen and
setup-disjoint. The incomplete targeted-v7 shards are preserved but excluded.

The existing deployment manifests and all prior checkpoints remain unchanged.
