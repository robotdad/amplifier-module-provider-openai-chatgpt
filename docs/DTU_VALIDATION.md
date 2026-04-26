# DTU Validation Guide

End-to-end validation of the ChatGPT provider module using a
[Digital Twin Universe](https://github.com/microsoft/amplifier-bundle-digital-twin-universe)
environment -- an isolated container running Amplifier against the live ChatGPT backend API.

## Why DTU Validation?

Unit tests (185+) validate internal logic in isolation. DTU validation proves the
module works as a real user would experience it: installed from git, configured via
`settings.yaml`, authenticated with a real OAuth token, and completing inference
against the live backend. It catches integration issues that unit tests cannot:

- Provider fails to mount inside a clean Amplifier installation
- OAuth token loading breaks with a different filesystem layout
- Routing matrix doesn't resolve agent roles
- SSE streaming fails against the actual backend
- Tool dispatch round-trip breaks end-to-end

## Prerequisites

1. **Incus** -- container runtime used by DTU
   ```bash
   incus version   # verify it's installed and running
   ```

2. **amplifier-digital-twin CLI**
   ```bash
   uv tool install git+https://github.com/microsoft/amplifier-bundle-digital-twin-universe
   amplifier-digital-twin --help
   ```

3. **A valid OAuth token** -- you need to have completed the device code login
   flow at least once on your host machine. The token file lives at:
   ```
   ~/.amplifier/openai-chatgpt-oauth.json
   ```
   If you don't have one, install the provider locally and run
   `amplifier provider add openai-chatgpt` to trigger the device code flow.

4. **Network access** -- the container needs to reach `chatgpt.com` and
   `auth.openai.com` for inference and token refresh.

## Quick Start

```bash
# Launch the environment
amplifier-digital-twin launch \
  .amplifier/digital-twin-universe/profiles/chatgpt-provider-reality-check.yaml \
  --var OAUTH_TOKEN_FILE=$HOME/.amplifier/openai-chatgpt-oauth.json

# Wait for readiness (CLI installed, token present, routing matrix present)
amplifier-digital-twin check-readiness <id>

# Run a quick smoke test
amplifier-digital-twin exec <id> -- amplifier run --mode single "What is 2+2?"

# Destroy when done
amplifier-digital-twin destroy <id>
```

Replace `<id>` with the DTU ID returned by the launch command (e.g. `dtu-8713bdb3`).

## What's in the Profile

The profile at `.amplifier/digital-twin-universe/profiles/chatgpt-provider-reality-check.yaml`
provisions an Ubuntu 24.04 container with:

| Component | How It Gets There |
|-----------|-------------------|
| Amplifier CLI | `uv tool install` from upstream git |
| ChatGPT provider module | `amplifier module add` + `provider install` from this repo's git URL |
| OAuth token | Copied from host via `--var OAUTH_TOKEN_FILE` (mode 0600) |
| Routing matrix | Copied from `routing/openai-chatgpt.yaml` in this repo (relative path) |
| Provider config | `settings.yaml` written with `login_on_mount: false` |

The `--var OAUTH_TOKEN_FILE` mechanism keeps the host-specific token path out of the
checked-in profile, making it portable across machines.

### Why login_on_mount: false?

The device code login flow requires an interactive browser. DTU containers are headless.
Instead, the profile copies a pre-authenticated token from the host and disables the
login prompt entirely. If the access token has expired but the refresh token is still
valid, the provider refreshes silently.

## Acceptance Tests

The acceptance test suite at `.amplifier/digital-twin-universe/acceptance-tests/chatgpt-provider.yaml`
defines 7 tests across 3 groups:

### Environment Setup

| Test | Validates |
|------|-----------|
| `env-amplifier-installed` | Amplifier CLI is installed and runnable |
| `env-token-present` | OAuth token file exists with mode 0600 |
| `env-routing-matrix-present` | Routing matrix file is installed |

### Provider Functionality

| Test | Validates |
|------|-----------|
| `provider-mounts` | Provider mounts without a device code prompt |
| `inference-text-completion` | Simple text completion works (2+2=4) |
| `inference-tool-use` | Tool dispatch round-trip through bash |
| `token-loaded-from-disk` | Token is read from the file path |

### Running Tests Manually

Each test command can be run inside the DTU:

```bash
# Environment checks
amplifier-digital-twin exec <id> -- amplifier --version
amplifier-digital-twin exec <id> -- stat -c '%a' /root/.amplifier/openai-chatgpt-oauth.json

# Inference
amplifier-digital-twin exec <id> -- amplifier run --mode single "What is 2+2? Reply with ONLY the number."

# Tool dispatch
amplifier-digital-twin exec <id> -- amplifier run --mode single "Run the command echo TOOL_DISPATCH_OK in bash and tell me the output"
```

### Running via Amplifier Reality Check

The acceptance tests are also compatible with the
[reality-check](https://github.com/microsoft/amplifier-bundle-reality-check) pipeline
for automated validation with structured reporting.

## What's NOT Tested

These are intentionally excluded and documented in the test file:

| Excluded Test | Reason |
|---------------|--------|
| OAuth device code login flow | Requires an interactive browser |
| OAuth token refresh (forced) | Risky to manufacture an expired token in a live environment |
| Model catalog name assertions | Model names change with subscription tier |

## Updating the Environment

The profile includes an `update` block that pulls the latest provider code
without a full rebuild:

```bash
amplifier-digital-twin update <id>
```

This clears the provider cache and reinstalls Amplifier. It does not re-copy
the OAuth token or routing matrix (those persist in the container).

## Troubleshooting

### Launch fails: "Incus not found"

Install Incus and ensure you're in the `incus-admin` group:
```bash
sudo usermod -aG incus-admin $USER
# Log out and back in (or: newgrp incus-admin)
incus list   # verify
```

### Readiness check fails: "token-present"

The OAuth token wasn't copied. Verify the source file exists on the host:
```bash
ls -la ~/.amplifier/openai-chatgpt-oauth.json
```
If missing, run the device code flow locally first:
```bash
amplifier provider add openai-chatgpt
```

### Inference fails: 401 Unauthorized

The OAuth token has expired and the refresh token is also invalid. Re-run the
device code flow on the host to get fresh tokens, then relaunch the DTU (the
token is copied at launch time, not dynamically).

### Inference fails: timeout

The ChatGPT backend can be slow under load. The acceptance tests use 120-180s
timeouts. If you're hitting them consistently, check network connectivity from
inside the container:
```bash
amplifier-digital-twin exec <id> -- curl -sI https://chatgpt.com
```

### Usage shows Input: 0 | Output: 0

Known issue. The ChatGPT backend SSE event shape doesn't include the same usage
fields as the standard OpenAI API. The provider's hook event emission doesn't
report token counts. Inference works correctly -- this is purely an accounting
gap in observability. Tracked for follow-up.