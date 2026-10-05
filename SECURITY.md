# Security policy

## Supported versions

There are no tagged releases yet. Security fixes are made on the `main` branch only.

| Version | Supported |
|---|---|
| `main` | Yes |
| anything older | No |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub instead: open the repository's **Security** tab and choose
**Report a vulnerability**. Only the maintainer can see the report.

Please include:

- what the problem is and which file, script or command it affects
- the steps to reproduce it, and the version (commit hash) you tested
- what an attacker could do with it

You can expect an acknowledgement within 7 days. Once the problem is confirmed, a fix will be
prepared on `main`, and you will be credited in the changelog unless you would rather not be.

## What is in scope

irsim is a simulator that runs locally. The most relevant problems are things like:

- a crafted config, YAML, scene, weather or asset file that runs code or writes outside the
  working directory when loaded
- a download script (`scripts/fetch_*.py`) that can be tricked into fetching or overwriting the
  wrong files
- secrets or credentials committed to the repository by mistake

Wrong physics, inaccurate images and crashes on bad input are ordinary bugs; please report those
with the [bug report form](https://github.com/reza-shahriari/isaac-thermal-camera-simulation/issues/new/choose).
