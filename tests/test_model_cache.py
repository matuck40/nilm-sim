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
    monkeypatch.setattr("nilm_sim.model_cache.code_fingerprint", lambda: "different")
    cached_fit("m", lambda: calls.append(1) or 2, cache_dir=tmp_path)
    assert len(calls) == 2


def test_fingerprint_is_stable_and_short():
    assert code_fingerprint() == code_fingerprint()
    assert len(code_fingerprint()) == 12
