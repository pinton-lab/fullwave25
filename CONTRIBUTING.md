# Contributing to Fullwave 2.5

Thank you for your interest in contributing to Fullwave 2.5.

## Branches and pull requests

- Create a branch named `TYPE/BRANCH_NAME`. `TYPE` is one of `feature`, `bugfix`, `hotfix`, `docs`, `refactor`, `release`, `test` or `experiment`; `BRANCH_NAME` says what the branch does.
- Base the branch on `develop` and keep it up to date with `develop`.
- Open the pull request against `develop`. After review, `develop` is merged into `main` for a release.
- Write clear, concise commit messages.

## Code

- Add tests for new features and bug fixes.
- Install the pre-commit hooks; `make install` does this for you:

  ```sh
  pre-commit install
  ```

- [Ruff](https://docs.astral.sh/ruff/) checks and formats the code before each commit. If a rule gets in the way, say so in your pull request.

## New to the tools?

- [Visual Studio Code documentation](https://code.visualstudio.com/docs) and [source control in VS Code](https://code.visualstudio.com/docs/sourcecontrol/overview)
- [Git tutorial (W3Schools)](https://www.w3schools.com/git/default.asp)
- [uv documentation](https://docs.astral.sh/uv/)
- [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
