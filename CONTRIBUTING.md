# Contributing to Fullwave 2.5

Thank you for your interest in contributing to Fullwave 2.5!

Please read the following guidelines to ensure a smooth contribution process.

- When developing something new, please create a new branch such as `TYPE/BRANCH_NAME`.
  - TYPE can be `feature`, `bugfix`, `hotfix`, `docs`, `refactor`, `release`, `test`, or `experiment`.
  - `BRANCH_NAME` should be descriptive of the feature or fix you are working on.
  - see also: [GitHub Branching Name Best Practices](https://dev.to/jps27cse/github-branching-name-best-practices-49ei)
- Please write clear and concise commit messages.
- Please keep your branch up to date with the main branch or develop branch.
  - we use GitLab Flow for Git branching
    - ![Branching strategie](https://media.geeksforgeeks.org/wp-content/uploads/20240223124532/image-243.webp)
    - ref: [Git branching strategies](https://www.geeksforgeeks.org/git/branching-strategies-in-git/)
- Please make a pull request if you want to add a new feature to the main branch.
- You need to make a pull request to develop branch first, and then to main branch after the code review.
- Please write tests for new features or bug fixes.
- Please use the pre-commit tool to keep the code clean. Pre-commit is installed when you use the make command to install `fullwave-python`.
  ```sh
  pre-commit install
  ```
- [Ruff](https://docs.astral.sh/ruff/) will check your code and suggest improvements before you commit.
  - Sometimes, however, the fix is unnecessary and cumbersome. Let Masashi know if you want to remove some coding rules.
