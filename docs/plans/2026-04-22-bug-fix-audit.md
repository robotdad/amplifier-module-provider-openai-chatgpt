# Bug Fix Plan: Post-Audit Remediation

> Generated from a full audit of the AI-generated provider module.
> 22 bugs found. 3 critical, 7 high, 12 medium/low.
> The unit tests pass because they encode the same misunderstandings as the code.

**Goal:** Fix all critical and high bugs so the module actually mounts, authenticates, and completes requests. Fix medium bugs that affect multi-turn conversations. Defer enhancements (retry, continuation, circuit breaking) to a follow-up.

## Group A: Showstoppers (blocks all functionality)

| Bug | File | Fix |
|-----|------|-----|
| `list_models()` sync | `provider.py:128` | Add `async` keyword |
| `parse_tool_calls()` wrong signature | `provider.py:140` | Accept `ChatResponse`, extract `.tool_calls` |
| `amplifier-core` missing from prod deps | `pyproject.toml` | Move to `[project] dependencies` |

## Group B: Auth (blocks authentication and token lifecycle)

| Bug | File | Fix |
|-----|------|-----|
| Blocking `urlopen` in async fns | `oauth.py` (4 sites) | Replace with `httpx.AsyncClient` |
| `account_id` lost on refresh | `oauth.py:196` | Extract from `id_token` JWT after refresh |
| `expires_at=""` on direct tokens | `oauth.py:543` | Compute from `expires_in` |
| Exception swallowed in mount | `__init__.py:62` | Log exception details with `exc_info=True` |

## Group C: Conversation (breaks multi-turn and tool use)

| Bug | File | Fix |
|-----|------|-----|
| Assistant messages use `input_text` | `provider.py:238` | Use `output_text` for assistant role |
| String tool content silently dropped | `provider.py:256` | Handle string content as `function_call_output` |
| `_ensure_valid_tokens()` ignores custom path | `provider.py:373` | Thread `self._token_file_path` through |

## Group D: Cleanup (correctness and hygiene)

| Bug | File | Fix |
|-----|------|-----|
| `self._client` never used, `close()` no-op | `provider.py` | Remove dead client field, simplify close() |
| Missing `"reasoning"` capability | `provider.py:125` | Add to `ProviderInfo.capabilities` |
| `_is_ssh_session()` dead code | `oauth.py:501` | Delete |
| `redact_secrets` imported unused | `provider.py:27` | Wire into raw event emission or remove |

## Deferred (not blocking first working session)

- No 401 retry / mid-session refresh (BUG-5) -- enhancement
- Missing `response.content_part.delta` handler (BUG-8) -- edge case
- `SSEError` instead of `LLMError` subtypes (BUG-13) -- polish
- `raw_events` unconditional population (BUG-15) -- optimization
- `response.incomplete` as hard error (BUG-16) -- enhancement
- No retry logic (BUG-20) -- enhancement
- Second+ system messages dropped (BUG-22) -- edge case
- TOCTOU on token file (BUG-19) -- minor security

## Test Updates Required

Existing tests encode wrong assumptions. Must update:
- `test_provider.py`: `list_models` tests need `await`, `parse_tool_calls` tests need `ChatResponse` input
- `test_oauth.py`: Tests using `urlopen` mocks need to mock `httpx.AsyncClient` instead
- All tests must pass after changes
