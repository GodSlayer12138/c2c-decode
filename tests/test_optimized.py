from types import SimpleNamespace
import pytest
from c2c_decode.optimized import OptimizedC2C


def test_scoped_logits_and_cached_replay_restore_on_failure():
    reference = OptimizedC2C.__new__(OptimizedC2C)
    calls = []
    def forward(**kwargs):
        calls.append(kwargs)
        return kwargs
    models = [SimpleNamespace(forward=forward), SimpleNamespace(forward=forward)]
    reference.model = SimpleNamespace(model_list=models, kv_cache_dict={"held": "cache"})
    cached = {0: "clean receiver", 1: "clean sender"}
    with pytest.raises(RuntimeError):
        with reference.last_logits(cached):
            assert models[0].forward() == "clean receiver"
            assert models[1].forward() == "clean sender"
            assert models[0].forward(input_ids="last token")["logits_to_keep"] == 1
            raise RuntimeError("injected")
    assert all(model.forward is forward for model in models)
    assert not reference.model.kv_cache_dict and not cached
    assert len(calls) == 1
