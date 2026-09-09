"""Embedded SQLCipher key for the public benchmark zipapp.

This module is compiled into ``novamind-operation`` at build time. The engine
inside the zipapp imports it via ``saas_bench.runtime._embedded_key`` to obtain the
key it needs to encrypt/decrypt ``world.nmdb``.

Humans should use the host-side decode script to inspect completed runs. The
key is intentionally not duplicated in documentation, but it remains
extractable from this Agent-visible zipapp; SQLCipher is therefore not the
formal isolation boundary. See ``docs/engineering/database-encryption.md``.

If you regenerate this key, every previously-recorded ``world.nmdb`` becomes
permanently undecryptable. Don't rotate without intent.
"""

_NMDB_KEY = "72692ea1293c52fbeef3ad8587db9d4fe3546d744766c3bdcb5aab4dad4b3c34"
