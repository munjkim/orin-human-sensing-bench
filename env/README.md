# Board environment snapshots

One directory per capture, produced by `./scripts/collect_env.sh` and
committed so board state can be reviewed without pasting terminal output.

```bash
./scripts/collect_env.sh
git add env && git commit -m "Add board environment snapshot" && git push
```

Each snapshot holds:

| file | what |
|---|---|
| `doctor.txt` / `doctor.json` | platform snapshot, prerequisite checks, reproducibility warnings |
| `jtop-schema.json` | one raw jtop payload — pins the power extractors to this jetson-stats version |
| `cameras.txt` / `cameras.json` | camera modes as the harness parses them |
| `v4l2-video*-formats.txt` | the raw `v4l2-ctl` output the parse came from |
| `nvpmodel.txt`, `jetson-clocks.txt` | power mode and clock state |
| `pip-freeze.txt` | the exact resolved environment |
| `META.txt` | when, which host, which commit |

Raw `v4l2-ctl` output is kept alongside the parsed version on purpose: when a
camera mode is missing from a benchmark, that pair is what distinguishes a
parser bug from a camera that genuinely does not offer the mode.

Every probe's failure is recorded in its own file rather than aborting the
run, so a snapshot from a board missing `v4l2-utils` or `jetson-stats` is
still worth reading.
