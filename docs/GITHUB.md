# Local Git and manual GitHub upload

The Desktop project is a local Git repository on branch `main`. The initial
commit contains the code and documentation. No remote is configured and nothing
has been published. Git uses your existing configured author identity.

## During experiments

Run these commands from the project root:

```bash
git status
git diff
git add lab scripts tests docs README.md pyproject.toml
git diff --cached
git commit -m "Describe the experiment or code change"
```

Only commit when you have an intentional change. Local virtual environments,
downloaded weights, caches, logs, secrets in `.env` files, and `results/` are
ignored. Keep the complete raw results in your local or cluster storage.
An ignore rule does not detect every secret: inspect staged changes before
committing, especially if you add configuration or credentials later.

## After the experiments

Create a curated, portable evidence folder:

```bash
mkdir -p docs/results
# Replace YOUR_RUN with the completed cluster run directory.
cp results/YOUR_RUN/report/REPORT.md docs/results/
cp results/YOUR_RUN/report/cells.csv docs/results/
cp results/YOUR_RUN/report/tradeoffs.png docs/results/
cp results/YOUR_RUN/environment.json docs/results/
```

Review the files for private paths or identifiers before committing. Add your
measured conclusions to the README and complete `docs/RESEARCH_NOTES.md`. Identify
the hardware, model, precision, request counts, repeats, and limitations. Label
Mac smoke measurements separately from cluster optimization comparisons.

```bash
git add docs/results docs/RESEARCH_NOTES.md README.md
git diff --cached
git commit -m "Add measured inference experiment results and analysis"
```

## Publish manually when ready

Create an empty repository in your GitHub account. Leave its initialization
options empty so it does not create a separate README or initial commit.
Choose your repository name, then replace BOTH placeholders below with your
actual GitHub username and repository name:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Use your existing GitHub authentication when prompted. If `origin` already
exists, inspect `git remote -v` and confirm its destination before changing it.
For later updates, commit locally and run `git push`.

The supplied source ZIP is for transferring code. Use the Desktop repository
for preserving this local Git history, or transfer its `.git` folder along with
the project if you need to work from another machine.
