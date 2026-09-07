# Mask template discriminator
- [x] External dispatch impact UNKNOWN; copied source callers traced. No production edits.
- [ ] Same reg kernel and causal semantics, change unused mask template float->bool with has_mask=false. Distinct pipeline identity, same existing metallib.
- [ ] Test numerical equivalence normal/stress at31/1055 and1024/32768; six interleaved rounds against currentquery128, regexplicit and originalregcausal.
- [ ] Restore server, inspect results plus GLM finding; promote nothing without demonstrated win.
