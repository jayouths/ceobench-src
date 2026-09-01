# SQLCipher Key for `world.nmdb`

This is the symmetric key used to encrypt and decrypt every CEO-Bench session
database. It is bundled into the published `novamind-operation` zipapp at
build time (compiled into `saas_bench.runtime._embedded_key._NMDB_KEY`) so the engine
can open the file without any external configuration.

## The key

```
_NMDB_KEY = "72692ea1293c52fbeef3ad8587db9d4fe3546d744766c3bdcb5aab4dad4b3c34"
```

64 hex characters. The engine passes this string verbatim as the SQLCipher
key — `PRAGMA key = '<hex_string>'` — so SQLCipher runs its default PBKDF2
derivation over the ASCII bytes of the hex string (it is **not** the
`PRAGMA key = "x'…'"` raw-bytes syntax). See
`src/saas_bench/runtime/db_protection.py::_apply_key`.

## Decrypting a session file

With the `sqlcipher` CLI installed:

```bash
sqlcipher path/to/run_<id>/world.nmdb
sqlcipher> PRAGMA key = '72692ea1293c52fbeef3ad8587db9d4fe3546d744766c3bdcb5aab4dad4b3c34';
sqlcipher> SELECT day, category, amount FROM ledger ORDER BY day, id LIMIT 10;
```

From Python:

```python
import sqlcipher3
conn = sqlcipher3.connect("path/to/world.nmdb")
conn.execute(
    "PRAGMA key = '72692ea1293c52fbeef3ad8587db9d4fe3546d744766c3bdcb5aab4dad4b3c34'"
)
for row in conn.execute("SELECT day, category, amount FROM ledger ORDER BY day, id"):
    print(row)
```

For the database schema and richer analysis recipes, see
[`analyze_trajectory.md`](analyze_trajectory.md).

## Rotation policy

**Don't rotate this key without intent.** Every previously recorded
`world.nmdb` is encrypted with this exact value; rotating it makes those
files permanently undecryptable. If you do rotate:

1. Update `src/saas_bench/runtime/_embedded_key.py::_NMDB_KEY`
2. Update the value above in this file
3. Rebuild the public zipapp (`uv run python scripts/build_public.py`) and
   push the new `public/` submodule pointer
4. Archive (or delete) all pre-rotation `outputs/runs/` artifacts so no
   one tries to decrypt them with the new key

## Why this lives in a plaintext file

The benchmark is client-side: the agent runs on the same machine as the
engine, so the key must be accessible to the engine process and is therefore
*physically* extractable by a determined adversary. We rely on (a) the
public README's explicit "do not inspect" rule, (b) the SQLCipher /
zipapp-bytecode obfuscation as a speed bump, and (c) the human evaluator
inspecting agent traces post-hoc. See the README's "⚠️ Caveat" section for
the full threat model.
