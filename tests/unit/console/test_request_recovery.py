"""Input rejection preserves the serial model owner; device faults still reset it."""
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hqsb.backends.interactive import InteractivePyTorch
from hqsb.console.app import PREFIX, create_app
from hqsb.console.config import DeploymentConfig, Settings
from hqsb.console.service import ConsoleService, ServiceError


class Actor:
    def __init__(self, _):
        self.closed = False

    def close(self):
        self.closed = True

    def execute(self, kind, payload, *_args):
        if kind == 'token_budget':
            return {'input_tokens':13, 'context_limit':256, 'max_output_tokens':243}
        if payload['max_output_tokens'] == 255:
            return {'finish_reason':'rejected', 'request_error':{'code':'CONTEXT_LIMIT','message':'13+255>256'},
                    'metrics':{'input_tokens':13, 'available_output_tokens':243, 'output_tokens':0}}
        return {'text':'a story', 'metrics':{'output_tokens':16}}


def wait(service, row):
    for _ in range(100):
        row = service.store.get(row['id'])
        if row['state'] in {'completed','failed'}:
            return row
        time.sleep(.01)
    raise AssertionError('request stuck')


def test_context_rejection_then_retry_keeps_epoch_and_actor(tmp_path):
    settings = Settings(data_dir=tmp_path/'state', evidence_root=tmp_path, web_dist=tmp_path/'dist',
        deployments=[DeploymentConfig(id='tiny',name='tiny',input_mode='completion',context_limit=256)])
    service = ConsoleService(settings, actor_factory=Actor)
    service.states['tiny'].update(state='ready',epoch='same-epoch')
    body = {'deployment_id':'tiny','expected_epoch':'same-epoch','prompt':'A story','max_output_tokens':255,'observation_mode':'off'}
    with TestClient(create_app(settings,service=service,monitor=False)) as client:
        headers = {'X-HQSB-Client':'console','Idempotency-Key':'too-long'}
        row = client.post(PREFIX+'/requests',json=body,headers=headers)
        assert row.status_code == 202
        failed = wait(service,row.json())
        assert failed['state'] == 'failed' and failed['cleanup'] == 'succeeded'
        assert failed['error_code'] == 'CONTEXT_LIMIT'
        assert not service.actors['tiny'].closed
        assert service.states['tiny']['epoch'] == 'same-epoch' and service.states['tiny']['state'] == 'ready'
        preview = client.post(PREFIX+'/requests/token-budget',json=body,headers=headers)
        assert preview.status_code == 200 and preview.json()['max_output_tokens'] == 243
        assert len(service.store.list()['items']) == 1, 'preview must not create an inference run'
        service.pending = 1
        with pytest.raises(ServiceError,match='稍后'):
            service.token_budget('tiny',body)
        service.pending = 0
        body['max_output_tokens'] = 16
        headers['Idempotency-Key'] = 'retry'
        retried = client.post(PREFIX+'/requests',json=body,headers=headers)
        assert retried.status_code == 202 and wait(service,retried.json())['state'] == 'completed'


def test_preview_uses_resident_tokenizer_without_forward():
    provider = InteractivePyTorch({'context_limit':256,'max_output_tokens':512})
    provider.model = SimpleNamespace(config=SimpleNamespace(model_type='llama'))
    provider.tokenizer = SimpleNamespace(encode=lambda text, **kw: list(range(13)))
    assert provider.token_budget({'prompt':'Once upon a time', 'input_mode':'completion'},None) == {
        'input_tokens':13,'context_limit':256,'max_output_tokens':243}


def test_preview_fault_invalidates_epoch_even_if_worker_cleanup_fails(tmp_path):
    settings = Settings(data_dir=tmp_path/'state', evidence_root=tmp_path, web_dist=tmp_path/'dist',
        deployments=[DeploymentConfig(id='tiny',name='tiny')])
    service = ConsoleService(settings,actor_factory=Actor)
    service.states['tiny'].update(state='ready',epoch='epoch')
    actor = service.actors['tiny']
    def fault(*_args):
        raise RuntimeError('controlled failure')
    original_close = actor.close
    actor.execute = actor.close = fault
    try:
        with pytest.raises(RuntimeError,match='controlled failure'):
            service.token_budget('tiny',{'expected_epoch':'epoch','messages':[{'role':'user','content':'x'}]})
        assert service.states['tiny']['state']=='failed' and service.states['tiny']['epoch'] is None
        assert service.pending == 0
    finally:
        actor.close = original_close
        service.close()
