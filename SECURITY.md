# Security

## Security model

qml-router stores conversation state, which can contain sensitive content. The isolation
results in the paper hold under these assumptions (paper Section VI-A):

| Assumption | Consequence |
|---|---|
| Identities are assigned by a trusted gateway. | The engine trusts the `session_id` and `user_id` in each request. Do not expose an engine port to untrusted clients; put the gateway in front of it. |
| The serving process runs inside a trusted host boundary. | The SQLite and LMDB stores are opened inside the engine process and are readable by any code running there. A compromise of the serving process can expose the session state that process can reach. |
| The identity key is not an authentication or authorization mechanism. | Authentication belongs to the gateway (API keys in `router/`). |

Isolation in this design means that the storage lookup has no cross-session content
matching step: state is addressed by session identity alone. It does not protect against
a wrong identity assigned by the gateway, authorization failures, stale metadata or a
corrupted identity mapping.

## Not provided in this release

| Item | Status |
|---|---|
| Encryption of stored state at rest | Not implemented (roadmap: per-user encryption) |
| Persister in a separate process | Not implemented (roadmap) |
| API-key checks on the legacy console routes of the gateway (`/api/chat` and `/api/chat_stream` without a key, `/api/sessions`, `/api/prewarm`, `/api/cmd_status` in `router/qml_webui.py`) | Missing in this release. The OpenAI-compatible route `/v1/chat/completions` and `/api/session_cmd` check the key and the session owner. Run the gateway only on a trusted network until this is fixed. |

## Reporting a vulnerability

Please report vulnerabilities privately to the maintainers rather than in a public issue.
Send the report to jun@kpst.co.kr.
