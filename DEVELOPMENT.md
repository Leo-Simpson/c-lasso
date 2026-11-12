# Development branches

- `master`: maintained R1–R4 and C1–C2 release baseline.
- `multi-class`: joint MC1/MC2 paths, metadata and tree constraints; no ridge API.
- `ridge`: fixed ridge for R1–R4/C1–C2, including concomitant perspectives.
- `quantile`: independent quantile regression development.

This checkout develops **ridge**. Both split branches include maintenance
baseline `d22da8d5ea54`. They share numerical techniques but neither needs the
other feature branch to run. Combining ridge with MC1/MC2 is a later integration
step; the original combined implementation and its tests remain in commit
`1e11ae3ede12`, an ancestor of both branches.

The split preserves history with ordinary merge commits; it does not rewrite
GitHub history. Local uncommitted work in other checkouts is left untouched.
