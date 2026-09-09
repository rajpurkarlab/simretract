"""Reference simulation statistics and first-attempt reporting contracts."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simretract.eval import driver, score

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / 'data' / 'evaluation'


def reference_summary():
    with (REFERENCE / 'simulation_summary.csv').open(newline='') as source:
        return list(csv.DictReader(source))


@pytest.mark.parametrize('expected', reference_summary(), ids=lambda r: r['preset'])
def test_paper_simulation_row_matches_first_attempt_records(expected):
    reference = json.loads((REFERENCE / 'simulation_first_run.json').read_text())
    assert reference['attempt'] == 'first'
    seeds, _ = score.load_seeds(REPO / reference['seed_selection'])
    rows = reference['models'][expected['preset']]
    assert len(rows) == len(seeds) == int(expected['n_trials']) == 50
    assert [r['seed'] for r in rows] == seeds
    result = score.funnel({r['seed']: r for r in rows}, seeds)
    assert result.n_results == 50 and not result.missing
    assert result.secured == int(expected['secured'])
    assert result.grasped_pct == float(expected['secured_percent'])
    for metric, prefix, unit in ((result.offset_mm, 'offset', 'mm'),
                                  (result.twist_deg, 'twist', 'deg')):
        assert metric.n == result.secured
        for field in ('mean', 'sd', 'median'):
            assert f'{getattr(metric, field):.1f}' == f'{float(expected[f"{prefix}_{field}_{unit}"]):.1f}'


def test_benchmark_defaults_and_explicit_retry_mode():
    parser = driver.build_parser()
    defaults = parser.parse_args([])
    assert defaults.passes == 1 and defaults.score_attempt == 'first'
    diagnostics = parser.parse_args(['--passes', '3', '--score-attempt', 'latest'])
    assert diagnostics.passes == 3 and diagnostics.score_attempt == 'latest'


def test_repeated_retry_invocations_preserve_first_result(tmp_path):
    current = tmp_path / 'result_7.json'
    first = {'seed': 7, 'outcome': 'timeout', 'queries': 9}
    current.write_text(json.dumps(first))
    assert driver.select_seeds(tmp_path, [7], 2) == [7]
    archive = tmp_path / 'result_7.try1.json'
    original = archive.read_bytes()
    current.write_text(json.dumps({'seed': 7, 'outcome': 'timeout', 'queries': 8}))
    assert driver.select_seeds(tmp_path, [7], 2) == [7]
    assert archive.read_bytes() == original
    assert (tmp_path / 'result_7.try2.json').is_file()
    assert score.load_results(tmp_path)[7] == first


@pytest.mark.parametrize('attempt,secured', [('first', 0), ('latest', 1)])
def test_driver_reports_selected_attempt(tmp_path, monkeypatch, attempt, secured):
    from simretract import config
    monkeypatch.setattr(config, 'require_kit_eula', lambda: None)
    monkeypatch.setattr(driver, 'Server', lambda *args: SimpleNamespace(stop=lambda: None))
    seeds = tmp_path / 'seeds.json'
    seeds.write_text(json.dumps({'name': 'fixture', 'seeds': [7]}))
    (tmp_path / 'result_7.try1.json').write_text(json.dumps({'seed': 7, 'outcome': 'timeout'}))
    (tmp_path / 'result_7.json').write_text(json.dumps({
        'seed': 7, 'outcome': 'completed', 'drop_mm': 0, 'held': False,
        'verdict': {'grasp_t_mm': 8, 'grasp_twist_deg': 2, 'grasp_r_deg': 3},
    }))
    cfg = driver.DriverConfig(ckpt=tmp_path/'checkpoint', out=tmp_path,
                              seeds_json=seeds, score_json=seeds, name='fixture',
                              score_attempt=attempt)
    result = driver.run(cfg)
    assert result.secured == secured
    saved = json.loads((tmp_path/'results.json').read_text())
    assert saved['score_attempt'] == attempt
    assert saved['statistics'] == {'population': 'secured', 'sd_ddof': 0}
    table = (tmp_path/'table.md').read_text()
    assert f'Attempt: {attempt}; SD: population.' in table
    assert 'Grasped (%)' in table


def test_completed_first_pass_can_be_extended_with_retry_diagnostics(tmp_path, monkeypatch):
    from simretract import config
    monkeypatch.setattr(config, 'require_kit_eula', lambda: None)
    server = SimpleNamespace(ensure=lambda: None, stop=lambda: None, alive=lambda: True)
    monkeypatch.setattr(driver, 'Server', lambda *args: server)
    seeds = tmp_path / 'seeds.json'
    seeds.write_text(json.dumps({'name': 'fixture', 'seeds': [7]}))
    current = tmp_path / 'result_7.json'
    current.write_text(json.dumps({'seed': 7, 'outcome': 'timeout'}))
    calls = []

    def worker(cfg, seed, log_path):
        calls.append(seed)
        current.write_text(json.dumps({'seed': seed, 'outcome': 'completed',
                                       'drop_mm': 0, 'held': False, 'verdict': {}}))
        return {'outcome': 'completed'}

    monkeypatch.setattr(driver, 'run_worker', worker)
    cfg = driver.DriverConfig(ckpt=tmp_path/'checkpoint', out=tmp_path,
                              seeds_json=seeds, score_json=seeds, name='fixture', passes=2)
    result = driver.run(cfg)
    assert calls == [7]
    assert result.secured == 0  # The first attempt remains the benchmark result.
    assert json.loads((tmp_path/'result_7.try1.json').read_text())['outcome'] == 'timeout'
    assert score.load_results(tmp_path, attempt='latest')[7]['outcome'] == 'completed'
