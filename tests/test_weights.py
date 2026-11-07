import pytest

from forcecrm.features import DEFAULT_WEIGHTS, is_custom, normalize_weights


def test_defaults_are_used_and_not_flagged_custom():
    assert normalize_weights(None) == DEFAULT_WEIGHTS
    assert not is_custom(normalize_weights({'activity': 65, 'geography': 20, 'legal_type': 10, 'age': 5}))


def test_weights_are_normalised_to_one():
    weights = normalize_weights({'activity': 50, 'geography': 50, 'legal_type': 0, 'age': 0})
    assert weights == {'activity': .5, 'geography': .5, 'legal_type': 0, 'age': 0}
    assert is_custom(weights)


@pytest.mark.parametrize('weights', [
    {'activity': 1, 'geography': 1, 'legal_type': 1},
    {'activity': 1, 'geography': 1, 'legal_type': 1, 'age': 1, 'extra': 1},
    {'activity': -1, 'geography': 1, 'legal_type': 1, 'age': 1},
    {'activity': 'a', 'geography': 1, 'legal_type': 1, 'age': 1},
    {'activity': True, 'geography': 1, 'legal_type': 1, 'age': 1},
    {'activity': float('nan'), 'geography': 1, 'legal_type': 1, 'age': 1},
    {'activity': 0, 'geography': 0, 'legal_type': 0, 'age': 0},
    [65, 20, 10, 5],
])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        normalize_weights(weights)


def test_api_rejects_invalid_weights():
    from fastapi.testclient import TestClient
    from forcecrm.server import app
    with TestClient(app) as client:
        for bad in ['not json', '{"activity": 1}', '{"activity":0,"geography":0,"legal_type":0,"age":0}']:
            response = client.post('/api/analyze', data={'demo': 'true', 'weights': bad}, headers={'X-ForceCRM': 'local'})
            assert response.status_code == 400
            assert 'Invalid weights' in response.json()['detail']
