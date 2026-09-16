"""stdlib-only acceptance check for canonical Session/SQLite persistence.

This check is intentionally independent from pytest/PySide6 so devctl can
validate persistence on a clean runtime machine. The full pytest regression
suite remains in tests/ and is run in development environments with the
project's ``[dev]`` extras installed.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

from infrastructure.storage.session_store import SessionStore


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="itc-session-check-") as raw_tmp:
        root = Path(raw_tmp)
        generated = root / "data" / "generated"
        generated.mkdir(parents=True, exist_ok=True)
        legacy = generated / "runtime_entities.json"
        legacy.write_text(
            '{"entities":{"server":[{"name":"legacy","quantity":1,"price":10}]}}',
            encoding="utf-8",
        )
        db = generated / "runtime.sqlite3"

        # F: one-time lossless/idempotent legacy import.
        store = SessionStore(db, legacy_runtime_path=legacy)
        expect(store.legacy_import_count() == 1, "legacy JSON must be imported exactly once")
        expect(store.global_entities()["server"][0]["name"] == "legacy", "global legacy data missing")
        source_id = store.active_session_id()
        expect(store.session_state()["entities"]["server"][0]["name"] == "legacy", "session legacy data missing")

        # A/D: partial pipeline survives restart and input changes preserve old results as stale.
        def seed_partial(state: dict) -> None:
            state["candidate_pools"]["technical"] = [{"id": "GA-1", "score": 1.0}]
            state["ga_results"]["technical"] = {"best": "GA-1"}
            state["ahp_results"]["technical"] = {"ranking": ["GA-1"]}
            store.mark_artifact_current(
                state,
                "ahp:technical",
                parameters={"weights": {"cost": 1.0}},
                provenance={"method": "ahp"},
            )

        store.mutate_active_state(seed_partial)
        before_restart = store.session_state()
        store = SessionStore(db, legacy_runtime_path=legacy)
        expect(store.active_session_id() == source_id, "restart must restore last active session")
        expect(store.session_state()["ga_results"] == before_restart["ga_results"], "GA state lost on restart")
        expect(store.session_state()["ahp_results"] == before_restart["ahp_results"], "AHP state lost on restart")

        def mutate_input(state: dict) -> None:
            state["entities"]["server"][0]["price"] = 11

        store.mutate_active_state(mutate_input, input_change=True)
        stale_state = store.session_state()
        expect(stale_state["ahp_results"]["technical"] == {"ranking": ["GA-1"]}, "stale result was deleted")
        expect(stale_state["derived_meta"]["ahp:technical"]["validity"] == "stale", "result must become stale")

        # E: immutable export payload is DB-owned and can be rematerialized after restart.
        payload = b"# immutable session export\nrevision-aware\n"
        artifact = store.record_export(
            kind="decision-report",
            format="md",
            name="decision-report.md",
            payload=payload,
            metadata={"scope": "technical"},
        )
        store = SessionStore(db, legacy_runtime_path=legacy)
        materialized = store.rematerialize_export(artifact.artifact_id, root / "restored")
        expect(materialized.read_bytes() == payload, "export payload cannot be rematerialized")

        # B: fork is a deep copy; future mutations do not leak to source.
        fork = store.duplicate_session(source_id, name="Fork")
        expect(fork.session_id != source_id, "fork must have a new identity")

        def mutate_fork(state: dict) -> None:
            state["entities"]["server"][0]["name"] = "fork-only"
            state["ga_results"]["technical"]["best"] = "FORK-GA"

        store.mutate_active_state(mutate_fork, input_change=True)
        expect(store.session_state()["entities"]["server"][0]["name"] == "fork-only", "fork mutation failed")
        store.switch_session(source_id)
        expect(store.session_state()["entities"]["server"][0]["name"] == "legacy", "fork mutation leaked into source")
        expect(store.session_state()["ga_results"]["technical"]["best"] == "GA-1", "nested fork state is not isolated")

        # C: global reusable data and existing sessions are explicitly decoupled.
        store.replace_global_entities({"server": [{"name": "global-new", "quantity": 1, "price": 20}]})
        expect(store.session_state()["entities"]["server"][0]["name"] == "legacy", "global mutation leaked into existing session")
        fresh = store.create_session("From global", context_type="experiment")
        expect(store.session_state()["entities"]["server"][0]["name"] == "global-new", "new session did not snapshot global data")

        def mutate_fresh(state: dict) -> None:
            state["entities"]["server"][0]["name"] = "session-new"

        store.mutate_active_state(mutate_fresh, input_change=True)
        expect(store.global_entities()["server"][0]["name"] == "global-new", "session mutation leaked into global data")

        # F again: changing legacy JSON after migration must not re-import/overwrite SQLite truth.
        legacy.write_text(
            '{"entities":{"server":[{"name":"changed-outside","quantity":9,"price":999}]}}',
            encoding="utf-8",
        )
        store = SessionStore(db, legacy_runtime_path=legacy)
        expect(store.legacy_import_count() == 1, "legacy import is not idempotent")
        expect(store.active_session_id() == fresh.session_id, "last-active session was not preserved")
        expect(store.session_state()["entities"]["server"][0]["name"] == "session-new", "legacy JSON overwrote SQLite truth")

        print("Session/SQLite stdlib acceptance: PASS")


if __name__ == "__main__":
    main()
