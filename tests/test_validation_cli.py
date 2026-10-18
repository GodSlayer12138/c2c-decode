"""A tolerance pass must not silently pass the strict-token CLI gate."""
import json
from unittest.mock import Mock
import pytest
from c2c_decode import evaluate


@pytest.mark.parametrize("strict", [False, True])
def test_strict_gate_keeps_report_and_closes_decoder(monkeypatch, tmp_path, strict):
    output = tmp_path / "report.json"
    argv = ["evaluate", "validate", "--output", str(output)]
    if strict:
        argv.append("--strict-tokens")
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(evaluate, "metadata", lambda args: {})
    monkeypatch.setattr(evaluate, "OfficialC2C", lambda path: object())
    decoder = Mock()
    monkeypatch.setattr(evaluate, "PagedDecoder", lambda *args, **kwargs: decoder)
    monkeypatch.setattr(evaluate, "validate", lambda *args: {
        "passed": True, "strict_token_equivalence": False})
    if strict:
        with pytest.raises(SystemExit, match="Strict token equivalence failed"):
            evaluate.main()
    else:
        evaluate.main()
    decoder.close.assert_called_once()
    assert json.loads(output.read_text())["validate"]["strict_token_equivalence"] is False
