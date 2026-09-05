# README and architecture companions

User scope: nanobot + Higgs local nightly; README combined maximum 1000 words; one-line opening, TL;DR, features, architecture, contribution ending. Offline HTML source-linked diagram per repository. No inference/runtime changes or public push.

- [x] Read current READMEs, contribution docs, paths and branch state.
- [x] Write concise landing pages and source-linked architecture manifests.
- [x] Generate offline HTML, add drift/path checks and lightweight CI.
- [x] Verify word counts, links, generator failure cases and HTML layout bounds (browser local-file preview blocked).
- [x] Preserve unrelated changes; summarize local branch state.

Validation: both generators --check passed; isolated source-change and missing-path probes failed as expected; all local links resolve. 612 README words, 842 words including both HTML companions. Browser policy blocked file URLs; no browser rendering claim. Runtime code and binaries untouched.
