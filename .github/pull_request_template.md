## What and why

<!-- One change per PR. Link the issue or steering-committee decision if there is one. -->

## Checklist

- [ ] Every commit is signed off (`git commit -s`)
- [ ] `scripts/build.sh && uv run pytest -n auto` passes (a new or changed check needs its mock fault and a test that fails without the fix)
- [ ] `docs/tests/` updated (if a test, its level or its spec clause changed)
- [ ] `CHANGELOG.md` updated (if users notice the change)
