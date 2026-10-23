# Contributing to qml-router

Thank you for your interest in qml-router.

## License of contributions and the CLA

qml-router is dual-licensed: AGPL-3.0-only (see [LICENSE](LICENSE)) or a commercial license
(see [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md)). To keep offering both, the copyright
holder, Korea Platform Service Technology Co., Inc., must be able to license every part of the
code under both terms.

For that reason a pull request can be merged only after its author has signed a contributor
license agreement (CLA). The CLA lets the copyright holder distribute the contribution under
AGPL-3.0-only and under the commercial license; the contributor keeps the copyright of the
contribution. Ask for the CLA at jun@kpst.co.kr before you open a pull request with code.
Issues and bug reports need no CLA.

## Reporting bugs and asking questions

Open an issue with:

- what you ran (engine and version, model and quantization, hardware, the command or request),
- what you expected and what happened, with the relevant log lines,
- the qml-router commit you used.

Report security problems privately as described in [SECURITY.md](SECURITY.md), not in an issue.

## Pull requests

- Keep a pull request to one change, and describe what it changes and why.
- Every new source file starts with an SPDX line (`SPDX-License-Identifier: AGPL-3.0-only`) and
  the copyright line used in the existing files.
- Do not include model files, datasets, credentials, host names or internal addresses.
- Run `scripts/check_public.sh` and the unit tests in `tests/unit/` before you submit.
- A change that affects a measured result states the conditions it was measured under (engine,
  model, hardware, settings) and keeps the earlier result for comparison.
- The files in `artifact/` are the results behind the paper's numbers; they are not edited by
  pull requests.
