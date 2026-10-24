# Save experiments and publish manually

Your Desktop project is a local Git repository on branch `main`. No remote is
configured and nothing has been published. Virtual environments, downloaded
weights, local caches, job logs, and `results/` are ignored.

## Save code changes

From the project root:

```bash
git status
git diff
git add lab scripts tests docs README.md QUICKSTART.md pyproject.toml
git diff --cached
git commit -m "Describe the code or experiment change"
```

Keep complete raw data on your server/local storage. Inspect staged files for
credentials and private identifiers before each commit.

## After the server experiment

```bash
mkdir -p docs/results
cp results/cuda-JOB_ID/report/REPORT.md docs/results/
cp results/cuda-JOB_ID/report/cells.csv docs/results/
cp results/cuda-JOB_ID/report/tradeoffs.png docs/results/
cp results/cuda-JOB_ID/report/data.json docs/results/
```

Replace JOB_ID with the completed job. Review summary metadata for private paths
or identifiers. Complete `docs/RESEARCH_NOTES.md` and update the README with
measured findings, hardware, model, settings, and limitations.

```bash
git add docs/results docs/RESEARCH_NOTES.md README.md
git diff --cached
git commit -m "Add measured inference results and analysis"
```

## Interactive animation

After obtaining real results, build a static HTML/CSS/JavaScript frontend under
`docs/` that reads `results/data.json`. It should work without a GPU or model
server for viewers. Use relative URLs such as `./results/data.json` so a project
website works under a repository subpath. See `VISUALIZATION.md` for the contract.

The frontend is not implemented yet. Once implemented, it can be published with
GitHub Pages from the repository's `docs/` folder, using the hosting settings
available for your repository. GitHub hosts the replay, not inference itself.
See [GitHub's publishing-source documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

## Push when ready

Create an empty GitHub repository without an automatically created README or
separate initial commit. Replace both placeholders with your actual values:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

If `origin` already exists, inspect `git remote -v` before changing it. For future
updates, commit locally and run `git push`. The supplied source ZIP excludes
Git history; transfer `.git` separately if you want to continue the same history
on your server. You can also bring results back and commit them in your Desktop
repository.
