# Experiments Directory

This directory keeps compact experiment records that are useful across conversations and machines.

## Keep

- `submission_log.csv`: confirmed online submissions and leaderboard results
- `probe_summary_*.csv`: short-probe outcomes and promotion decisions
- `source_match_*.csv`: static source-matching tables used to gate new data sources
- small text manifests that define a selected subset used by a meaningful experiment

## Usually Ignore

- `*_rank_*.csv`
- `*_top1024*.txt`
- router scratch directories

These are rebuildable caches and are already covered by `.gitignore` when untracked.

## Naming Convention

- `probe_summary_YYYYMMDD.csv`: human-sized daily summary table
- `source_match_<source>_YYYYMMDD.csv`: static matching result for a candidate source
- `submission_log.csv`: canonical leaderboard record
