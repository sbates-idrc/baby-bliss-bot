# Contributing

## Process/Workflow

The project's [source code](https://github.com/inclusive-design/baby-bliss-bot) is hosted on GitHub. All of the code
that is included in a release lives in the main branch. The intention is that the main branch is always in a working
state.

This project uses a workflow where contributors fork the project repository, work in a branch created off of main,
and submit a Pull Request against the project repo's main branch.

### Issue Tracker

Issues are tracked using [GitHub Issues](https://github.com/inclusive-design/baby-bliss-bot/issues). When creating a
new issue, please pick the most appropriate type (e.g. bug, feature) and fill out the template with all the
necessary information. Issues should be meaningful and describe the task, bug, or feature in a way that can be
understood by the community. Opaque or general descriptions should be avoided. If you have a large task that will
involve a number of substantial commits, consider breaking it up into subtasks.

### Linting

In order to avoid errors and common pitfalls in the python language, all code should be regularly checked using the
provided lint task.

```bash
flake8
```

### Pull Requests and Merging

If you are starting work on a new feature or bug fix, create a new branch from 
[main](https://github.com/inclusive-design/baby-bliss-bot):

```bash
git checkout main
git checkout -b your-branch-name
```

Give your branch a descriptive name:

- For a new feature, call it `feat/description-of-feature`
- For a bug fix, call it `fix/description-of-bug`

When committing your changes, use [Conventional Commits](https://conventionalcommits.org/).

When your work is complete, open a pull request against the [main](https://github.com/inclusive-design/baby-bliss-bot) branch:

- The title of the pull request is in the format of `<type>: <description> (resolves <issue-id>)`
  - For example, for a new feature that resolves the issue id #1, the title is `feat: description of the feature (resolves #1)`. 
  This makes sure the issue id(s) is included in the commit history for easy access in the future.
- Please make sure to fill out the pull request template.

After a Pull Request (PR) has been submitted, one or more team members will review the contribution. This
typically results in a back and forth conversation and modifications to the PR. Merging into the project repo is a
manual process and requires at least one Maintainer to sign off on the PR and merge it into the project repo.

When merging a Pull Request, it is recommended to use a [Squash Merge](
https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-merges#squash-and-merge-your-pull-request-commits).
While this does modify commit history, it will enable us to more easily establish a link between code changes and Issues
that precipitated them.
