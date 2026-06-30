# Reverting the Shiraz fork migration

On **2026-06-30** this checkout (`/home/user/axol`, the **ZED box** `GTW-ONX1-C1D3MOD4`)
was migrated from the vendor SDK to the personal fork. This file documents exactly
what changed and how to put everything back the way it was — code, venv, and the
systemd service — at any granularity from "one piece" to "full restore".

> TL;DR fastest full revert: re-enable the service and swap the directory back
> from the untouched backup. See [§5 Full restore](#5-full-restore-nuclear-option).

---

## What the migration changed

| Area | Before (original) | After (migration) |
|------|-------------------|-------------------|
| `origin` remote | `https://github.com/almond-bot/axol.git` (vendor) | `https://github.com/mojtabamozaffar/axol.git` (fork) |
| `upstream` remote | *(did not exist)* | `https://github.com/almond-bot/axol.git` (vendor) |
| Checked-out branch | `docs/rtc-plan` @ `dbc9983` (+ 1 uncommitted edit) | `shiraz-main` @ `c7cb995` |
| Local `main` | `46b2b52` (old base + 4 personal commits) | `aeadfc5` (clean mirror of vendor, tracks `upstream/main`) |
| Personal work | 4 commits on old base + uncommitted edit + RTC doc | all rebased onto latest vendor as `shiraz-main` |
| `axol` service | `enabled` + `active` (vendor install at `/opt/axol/...`) | *(to be)* disabled + stopped; run from fork via `uv run axol` |
| venv | `almond-axol==0.1.1` | `almond-axol==0.1.2` (+ `pyzed`/gst reinstalled) |

### Safety nets created during migration (don't delete until you're sure)

- **Full repo backup:** `~/axol-shiraz` — a complete `cp -a` of the original
  `/home/user/axol` (including `.git` and `.venv`), taken **before any change**.
  It sits on `docs/rtc-plan` @ `dbc9983` with the original uncommitted
  `rerun_robot.py` edit, and its `origin` still points at the vendor. This is the
  truest "original".
- **Backup branches on the fork** (`origin`):
  - `backup/shiraz-main-orig-d60d56a` — the fork's original `shiraz-main` (`d60d56a`)
  - `backup/zedbox-main-20260630` — this box's original `main` (`46b2b52`)
  - `backup/zedbox-rtc-plan-20260630` — this box's original `docs/rtc-plan` (`dbc9983`)

---

## 1. Revert the services (most important on the ZED box)

Re-enable and start the vendor service exactly as it was (`enabled` + `active`):

```bash
sudo systemctl enable --now axol          # enable on boot + start now
systemctl status axol --no-pager          # confirm: active (running), enabled
```

The vendor unit (`/etc/systemd/system/axol.service`) runs
`/usr/local/bin/axol serve` from the self-updating install at
`/opt/axol/uv/tools/almond-axol` — it is independent of this checkout, so simply
re-enabling it restores the original behavior (including the vendor auto-updater).

If you created any custom user unit to run `serve` from the fork, remove it first:

```bash
# only if you added one
systemctl --user disable --now axol-fork 2>/dev/null || true
sudo systemctl disable --now axol-fork  2>/dev/null || true
```

> Stop using the fork before re-enabling the service: if you have `uv run axol serve`
> running from `/home/user/axol`, Ctrl-C it so it doesn't fight the service for port 8000.

---

## 2. Revert the git remotes and branches (in place)

This undoes the remote rename and restores the original checkout state without
deleting your fork work (it stays on the fork and in the backup branches).

```bash
cd /home/user/axol

# 2a. Remotes: fork -> remove, vendor -> origin again
git remote remove origin
git remote rename upstream origin
git fetch origin --prune

# 2b. Restore local main to its original tip (old base + 4 personal commits)
git switch main
git reset --hard 46b2b52

# 2c. Restore the original checked-out branch + its uncommitted edit
git switch docs/rtc-plan        # already at dbc9983 locally; if missing, recreate:
# git switch -c docs/rtc-plan 46b2b52   # then re-apply the RTC doc commit if needed

git remote -v                   # expect: origin = almond-bot/axol, no upstream
git branch -vv                  # expect: main @ 46b2b52
```

> The original `docs/rtc-plan` also had **one uncommitted edit** to
> `almond_axol/lerobot/rerun_robot.py` (translucent mesh / marker sizing). It was
> committed during migration as `c7cb995`. To reproduce the original *uncommitted*
> state, restore that file from the backup:
> `cp ~/axol-shiraz/almond_axol/lerobot/rerun_robot.py almond_axol/lerobot/rerun_robot.py`

If you also want to drop the migration-created branch locally:

```bash
git branch -D shiraz-main       # safe: still on the fork as origin/shiraz-main + backup
```

---

## 3. Revert the venv

The original venv had `almond-axol==0.1.1`. Restoring `main` to `46b2b52` (§2b)
and re-syncing rebuilds it. The ZED hardware wheels are **not** in the lockfile, so
reinstall them afterward (every `uv sync` prunes them):

```bash
cd /home/user/axol
uv sync --extra lerobot --extra sim          # rebuild editable install for this branch
uv run axol zed.install                      # restore pyzed (ZED SDK wheel)
uv run axol gst.install                       # restore GStreamer + PyGObject
# or, manual equivalent of the wheel restore:
# uv pip install ~/.almond/wheels/pyzed-5.2-cp313-cp313-linux_aarch64.whl \
#                "pygobject==3.50.2" "pycairo==1.29.0" "cython==3.2.6"
```

Tip: use `uv sync --inexact` to avoid pruning `pyzed`/gst in the first place.

---

## 4. (Optional) Clean up the fork-side artifacts

Only do this once you're certain you won't need the migrated work again. These are
on the fork (`origin`); deleting them is irreversible from here:

```bash
git push origin --delete shiraz-main                          # the migrated work branch
git push origin --delete backup/shiraz-main-orig-d60d56a
git push origin --delete backup/zedbox-main-20260630
git push origin --delete backup/zedbox-rtc-plan-20260630
```

Leave them in place if there's any chance you'll want the integrated work back.

---

## 5. Full restore (nuclear option)

If anything above is uncertain, the cleanest revert is to swap the whole directory
back to the pre-migration snapshot. The backup includes the original `.git`
(remotes, branches) and `.venv`, so this restores code **and** environment at once.

```bash
# 1. stop using the fork
#    (Ctrl-C any `uv run axol ...` running from /home/user/axol)

# 2. swap the directory
mv /home/user/axol /home/user/axol-migrated-$(date +%Y%m%d)   # set aside, don't delete yet
mv /home/user/axol-shiraz /home/user/axol                     # restore original

# 3. restore the service
sudo systemctl enable --now axol

# 4. sanity check
cd /home/user/axol
git remote -v        # origin = almond-bot/axol
git branch -vv       # docs/rtc-plan checked out, main @ 46b2b52
git status           # the original uncommitted rerun_robot.py edit is present
```

`/usr/local/bin/axol` (the vendor system install) is never touched by any of this,
so the service path keeps working throughout.

After verifying, you can remove the set-aside copy: `rm -rf /home/user/axol-migrated-*`.

---

## Quick reference — key commits & paths

```
fork (origin)      https://github.com/mojtabamozaffar/axol.git
vendor (upstream)  https://github.com/almond-bot/axol.git
vendor main        aeadfc5
original local main 46b2b52      (backup/zedbox-main-20260630)
original rtc-plan  dbc9983       (backup/zedbox-rtc-plan-20260630)
fork orig shiraz   d60d56a       (backup/shiraz-main-orig-d60d56a)
migrated shiraz    c7cb995
full repo backup   ~/axol-shiraz
ZED hardware wheel ~/.almond/wheels/pyzed-5.2-cp313-cp313-linux_aarch64.whl
vendor service     systemd unit `axol` -> /usr/local/bin/axol serve (/opt/axol/uv/tools/almond-axol)
```
