# Spec-Driven Development — Task Tracking

## How This Works

Each team member has their own task file (`member_a.md`, `member_b.md`, `member_c.md`) that contains every task they own, broken into daily checkpoints.

### Progress Markers

Update the status of each task as you work:

| Marker | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Done |
| `[!]` | Blocked (add reason) |

### Daily Workflow

1. Pull latest from `develop`.
2. Open your spec file (`specs/member_X.md`).
3. Check what's next for today.
4. Code it. Test it locally.
5. When a task is done, mark `[x]` and note the output artifact.
6. Commit the spec update alongside your code: `git add specs/ && git commit -m "spec: mark task X done"`
7. Push and PR.

### Checkpoints

Each day ends with a **checkpoint** — a concrete deliverable that the team verifies before moving on. These are listed in `checkpoints.md`.

### Files

```
specs/
├── README.md               ← You are here
├── checkpoints.md          ← Daily team checkpoints
├── member_a_data.md        ← Member A task list
├── member_b_aiml.md        ← Member B task list
└── member_c_backend.md     ← Member C task list
```
