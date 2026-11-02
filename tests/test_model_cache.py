from nilm_sim.model_cache import cached_fit, code_fingerprint


def test_second_call_loads_from_disk_without_training(tmp_path):
    calls = []

    def train():
        calls.append(1)
        return {"w": [1.0, 2.0]}

    first = cached_fit("m", train, cache_dir=tmp_path)
    second = cached_fit("m", train, cache_dir=tmp_path)
    assert first == second == {"w": [1.0, 2.0]}
    assert len(calls) == 1
    assert len(list(tmp_path.glob("m-*.joblib"))) == 1


def test_changed_code_trains_again(tmp_path, monkeypatch):
    calls = []
    cached_fit("m", lambda: calls.append(1) or 1, cache_dir=tmp_path)
    monkeypatch.setattr("nilm_sim.model_cache.code_fingerprint", lambda *files: "different")
    cached_fit("m", lambda: calls.append(1) or 2, cache_dir=tmp_path)
    assert len(calls) == 2


def test_fingerprint_is_stable_and_short():
    assert code_fingerprint() == code_fingerprint()
    assert len(code_fingerprint()) == 12


def test_corrupt_file_is_retrained_not_fatal(tmp_path):
    cached_fit("m", lambda: 1, cache_dir=tmp_path)
    (path,) = tmp_path.glob("m-*.joblib")
    path.write_bytes(b"not a model")
    assert cached_fit("m", lambda: 2, cache_dir=tmp_path) == 2
    assert cached_fit("m", lambda: 3, cache_dir=tmp_path) == 2      # the rewritten file loads
    assert not list(tmp_path.glob("*.tmp"))


def test_training_settings_outside_the_package_change_the_fingerprint(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("N_HOUSES = 10\n")
    b.write_text("N_HOUSES = 12\n")
    assert code_fingerprint(a) != code_fingerprint(b)
