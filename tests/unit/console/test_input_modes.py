"""Input semantics and template provenance; no GPU inference claims."""
import time

import pytest
from fastapi.testclient import TestClient

from hqsb.backends.prompting import prepare_prompt
from hqsb.console.app import PREFIX, InferenceRequest, create_app
from hqsb.console.config import DeploymentConfig, Settings
from hqsb.console.service import ConsoleService
from hqsb.experiments.benchmark_suite import suite


class Tokenizer:
    def apply_chat_template(self, messages, **options):
        self.messages, self.options = messages, options
        return '<publisher-template>'


def test_completion_preserves_raw_text_and_native_special_tokens_without_template():
    raw = '\n春天来了，  '
    assert prepare_prompt(object(), {'prompt': raw, 'input_mode': 'completion'}, 'gemma3_text') == (raw, True)
    tokenizer = Tokenizer()
    messages = [{'role': 'user', 'content': 'hello'}]
    assert prepare_prompt(tokenizer, {'messages': messages}, 'qwen3') == ('<publisher-template>', False)
    assert tokenizer.messages is messages
    assert tokenizer.options == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
    prepare_prompt(tokenizer, {'messages': messages}, 'gemma3_text')
    assert 'enable_thinking' not in tokenizer.options


@pytest.mark.parametrize('changes', [{}, {'prompt': 'x', 'messages': [{'role':'user','content':'y'}]}, {'prompt':''}])
def test_request_requires_exactly_one_input_format(changes):
    with pytest.raises(ValueError):
        InferenceRequest(deployment_id='base', expected_epoch='epoch', **changes)


def test_completion_suite_has_distinct_frozen_input_identity():
    chat, raw = suite(), suite('completion')
    assert chat['version'] != raw['version'] and chat['suite_sha256'] != raw['suite_sha256']
    for a, b in zip(chat['cases'], raw['cases']):
        assert b['messages'] == [] and b['input_mode'] == 'completion'
        assert b['prompt'] == a['prompt'] and b['input_sha256'] != a['input_sha256']


def test_bf16_deployment_keeps_fp16_only_custom_profiles_unavailable():
    from hqsb.console.kernel_inventory import deployment_catalogue
    config = DeploymentConfig(id='base', name='base', precision='bfloat16')
    catalog = deployment_catalogue(config)
    native = next(row for row in catalog['profiles'] if row['profile_id'] == 'framework-native')
    assert native['availability']['loadable']
    assert all(not row['availability']['loadable'] for row in catalog['profiles'] if row['operators'])
    with pytest.raises(ValueError, match='require FP16'):
        DeploymentConfig(id='base', name='base', precision='bfloat16', kernel_profile='qwen3-norm-vector-v2')


class Actor:
    def __init__(self, _):
        self.requests = []

    def close(self):
        pass

    def execute(self, kind, payload, on_output, is_cancelled, timeout_s):
        self.requests.append(payload)
        return {'text': 'continued', 'metrics': {'input_mode': payload.get('input_mode')}}


def test_raw_request_redacts_prompt_and_rejects_chat_on_base_model(tmp_path):
    settings = Settings(data_dir=tmp_path/'state', evidence_root=tmp_path, web_dist=tmp_path/'dist',
        deployments=[DeploymentConfig(id='base', name='base', input_mode='completion')])
    service = ConsoleService(settings, actor_factory=Actor)
    service.states['base'].update(state='ready', epoch='epoch')
    app = create_app(settings, service=service, monitor=False)
    headers = {'X-HQSB-Client':'console', 'Idempotency-Key':'raw-private'}
    with TestClient(app) as client:
        body = {'deployment_id':'base','expected_epoch':'epoch','prompt':'private raw content', 'max_output_tokens':8}
        response = client.post(PREFIX+'/requests',json=body,headers=headers)
        assert response.status_code == 202, response.text
        identity = response.json()['id']
        for _ in range(100):
            row = client.get(PREFIX+'/runs/'+identity).json()
            if row['state'] == 'completed':
                break
            time.sleep(.01)
        assert row['state'] == 'completed'
        assert row['config']['input_mode'] == 'completion' and 'prompt' not in row['config']
        assert len(row['config']['input_sha256']) == 64
        assert row['config']['generation_policy']['chat_template_owner'] == 'none_raw_text'
        assert service.actors['base'].requests[-1]['prompt'] == body['prompt']
        public = client.get(PREFIX+'/runs').json()['items'][0]
        assert 'prompt' not in public['config']
        body.update(save_input=True)
        headers['Idempotency-Key'] = 'raw-saved'
        saved = client.post(PREFIX+'/requests',json=body,headers=headers)
        assert saved.status_code == 202, saved.text
        assert saved.json()['config']['prompt'] == body['prompt']
        body.pop('prompt')
        body['messages'] = [{'role':'user','content':'hello'}]
        headers['Idempotency-Key'] = 'raw-cannot-chat'
        rejected = client.post(PREFIX+'/requests',json=body,headers=headers)
        assert rejected.status_code == 422 and rejected.json()['error']['code'] == 'INPUT_MODE'
        catalog = client.get(PREFIX+'/benchmark-catalog?deployment_id=base').json()
        assert catalog['input_mode'] == 'completion'
        assert all(case['input_mode'] == 'completion' for case in catalog['cases'])
