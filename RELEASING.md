# Releasing c-lasso

The next maintenance release is **1.0.12**. Multiclass MC1/MC2 support is planned
for **1.1.0**, optionally preceded by **1.1.0rc1**.

## One-time publishing setup (PyPI project owner)

In https://pypi.org/manage/project/c-lasso/settings/publishing/ add a GitHub
Trusted Publisher with these exact values:

| Field | Value |
| --- | --- |
| Owner | `bio-datascience` |
| Repository | `c-lasso` |
| Workflow filename | `python-publish.yml` |
| Environment | `pypi` |

Create the matching `pypi` environment in GitHub repository settings. Restrict
publishing to release tags and configure a required reviewer if desired. The
workflow requests an OIDC identity only in its publishing job; it needs no stored
PyPI username, password, or API token. Repository transfer does not by itself
transfer PyPI project ownership. Retire obsolete publishing secrets after the
new publisher is configured successfully.

## Validate and release

1. Merge the maintenance fixes and release preparation PR. Check both the Test
   and Release workflows on the intended master commit. The Test `build` gate
   requires every supported compatibility job; see `COMPATIBILITY.md`.
2. Review `CHANGELOG.md`, replace the unreleased heading with the release date,
   and merge that change before tagging.
3. Fetch master and all tags. Versioneer derives package versions from Git tags;
   do not edit `classo/_version.py` or hard-code a version in `setup.py`.
4. Create an annotated tag named exactly `1.0.12` on the tested commit and push
   that tag. Existing tags have no `v` prefix. Confirm that this version is not
   already present on PyPI. Do not move a tag after publishing it.
5. Create a GitHub release targeting that existing tag, using the corresponding
   changelog section as its notes. A **draft** does not publish to PyPI; publishing
   the GitHub release triggers the Release workflow.
6. The workflow runs tests, builds both distributions (the wheel is built from
   the source distribution), checks metadata, and installs the wheel into a clean
   environment outside the checkout. For release events, its version must equal
   the release tag. Release publication also reruns the complete compatibility
   matrix on that commit. Both must pass before publishing is enabled.
7. Verify https://pypi.org/project/c-lasso/1.0.12/ and install `c-lasso==1.0.12`
   in a fresh environment. Rebuild Read the Docs after confirming its repository
   connection points to `bio-datascience/c-lasso`.

To inspect distributions before creating a release, run the Release workflow
manually. Manual runs and pull requests build and validate only; they cannot
publish. Download the `distributions` artifact from the workflow run.

For 1.1.0, merge the supported multiclass interface, numerical tests, and Gallery
examples first. Repeat the same procedure using `1.1.0rc1` for an optional PyPI
prerelease and `1.1.0` for the final release. Mark the GitHub RC as a prerelease.
