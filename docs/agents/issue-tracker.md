# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files under `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues use one file per ticket:
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`
- A `Status:` line records the ticket state.
- Comments append under `## Comments`.

## Publishing

When a skill publishes an issue, create a new file under the corresponding `.scratch/<feature-slug>/` directory.

## Fetching

Read the ticket path or issue number supplied by the user.

## Wayfinder

- Map: `.scratch/<effort>/map.md`
- Child ticket: `.scratch/<effort>/issues/NN-<slug>.md`
- A ticket is unblocked when every item in `Blocked by:` is resolved.
