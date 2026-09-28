# 0159 — Sensor files never go stale: optional fields, migrators only for renames, an extensions block

Date: 2026-09-28
**Status:** Accepted
Roadmap: SC.32 (§12.2); builds on ADR 0008 (the loader and its hashes)

## Context

The owner's requirement (2026-09-28): a set of ready-made cameras, and a new camera must mean
writing one YAML file that the IR camera loads by name, with no Python; and "as we may add many
different things later to these yaml files ... they should be extendable that way even after 10
years of development and old yaml will work". The sensor schema is at v12 and has been additive
since v8, which is the right instinct but was never written down as a rule, and nothing tested
that an old file still loads to the same camera.

## Options considered

1. **Accept anything.** `extra="ignore"` on the schema: an old file loads, and so does a typo,
   silently. A misspelled `f_numbr` would render a plausible image with the default F-number,
   which is the failure mode this project exists to prevent.
2. **Version everything, migrate everything.** Bump `schema_version` on every new field and ship a
   migrator each time. Correct and heavy: a field with a default that *is* the old behaviour needs
   no migration, and a rule that demands one anyway will be skipped.
3. **Additive by default, a migrator only for a break, and one escape hatch.** The rule below.

## Decision

Option 3, in `irsim.config.catalogue` and `irsim.config.sensor`:

* **A new field is optional, and its default is the pre-existing behaviour.** Adding it changes
  no file's meaning and needs no version bump; the version-history comment at the top of the
  schema records what each field means for older files.
* **`schema_version` moves only for a breaking change** -- a rename, a unit change, a field whose
  default cannot be the old behaviour -- and then ships a migrator registered in
  `catalogue.MIGRATIONS[from_version]`. The loader walks a file from its version to today's, one
  step at a time, before validation. Every version from `MIN_SCHEMA_VERSION` up stays readable;
  the floor rises only when a version genuinely cannot be migrated.
* **`sensor.extensions:` is the one place a stranger may write.** It is carried through load and
  dump untouched and never validated: a serial number, a mount, a downstream tool's settings.
  Everywhere else `extra="forbid"` stays: forward compatibility is not silence about typos.
* **Frozen fixtures prove it.** `tests/fixtures/sensors/boson_v8..v11.yaml` are the Boson as
  written at each past version and are never edited; the fast tier loads each and compares it
  to today's file minus what later versions added. A change that breaks a fixture needs a
  migrator, not an edit to the fixture.
* **Names, not paths.** `--sensor boson640` resolves through `configs/sensor_catalogue.yaml`
  (data, not code); a file's stem always works; `$IRSIM_SENSOR_DIR` is searched first and
  shadows shipped cameras by file name, so a user's camera is a file they add, not an edit.

## Consequences

* The ten-year promise is a test, not an intention: every future version adds a fixture.
* A migrator is a plain function on the parsed document, testable without a camera.
* Calibration (SC.33) will be an optional `calibration:` block under this rule: absent, the
  camera is synthesised as today, bit for bit.
