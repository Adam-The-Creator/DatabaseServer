# Feedback module

Questionnaires (SUS, UEQ, a smiley rating for children), feedback messages and the admin dashboard of the DataAnalysis
web application. Requirements: `DataAnalysis/requirements.md`, phase P3 (FBK-*, ADM-*, SEC-*, TEC-13..16).

An additive module (SC-08): its own tables, its own routers, nothing else of the existing server changes except what
SC-08 lists (the Admin role, the sign-up check, the request throttling).

| File | What it does |
| ---- | ------------ |
| `instruments.py` | The questionnaires: item IDs, scales, validation, scoring. The only place with the scoring. |
| `store.py` | The three tables, and the rules of a submission: limits, idempotency, caps, text checks, pseudonym code. |
| `results.py` | What the admin sees: usage numbers, statistics, message list, CSV exports. |
| `../routers/feedback.py` | `/feedback/*`: instruments, send a questionnaire, send a message, own submissions. |
| `../routers/admin.py` | `/admin/*`: usage, results, messages, exports. Admin token only. |
| `../throttle.py` | Request limits per address and per account, login block, body size cap, security headers. |
| `../admin_account.py` | Creation of the admin account. |

## The admin account

The sign-up endpoint cannot create an admin. On the server PC:

```
python server.py --create-admin NAME          # asks for the password
set ADMIN_PASSWORD=...  &  python server.py --create-admin NAME     # or from the environment
```

The packaged exe takes the same argument. Running it again for an existing admin sets a new password; an account of
another role is never converted. An admin token is refused by every endpoint except `/admin/*` and the admin's own
`/players/{id}/role` and `/players/{id}/info`.

## Limits (config.py)

Send limits per account (`FEEDBACK_*`): a questionnaire (same instrument and target) once per hour, a message once per
minute and 20 per day, at most 500 messages and 200 questionnaires stored per account. They are counted from the stored
submissions, in the same write transaction as the insert. Request limits (`THROTTLE_*`, `LOGIN_*`) are in memory.
Behind a tunnel the client address is read from `X-Forwarded-For` only when the request comes from `TRUSTED_PROXIES`.

## Privacy

The tables keep the account ID of the sender (needed for the limits and for the user's own list). The admin API only
returns a random pseudonym code per account, role and age band at the time of sending, never the username, the real
name or the account ID. Someone with direct access to the database can still identify a sender.

## Tests

`python -m unittest discover -s tests -t .` from `DatabaseServer/` (`tests/test_feedback.py`).
