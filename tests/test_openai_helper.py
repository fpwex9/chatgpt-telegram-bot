import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'bot'))

import openai_helper  # noqa: E402
from openai_helper import OpenAIHelper, chat_completion_params, default_max_tokens, \
    are_functions_available, is_legacy_model  # noqa: E402

OLD_MODELS = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-32k', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'o1']
NEW_MODELS = ['gpt-6-luna', 'gpt-6-sol', 'gpt-5.5', 'gpt-5.4-mini', 'some-future-model']

CONFIG = {
    'api_key': 'test', 'temperature': 1.0, 'presence_penalty': 0.0, 'frequency_penalty': 0.0,
    'assistant_prompt': 'You are a helpful assistant.', 'vision_model': 'gpt-4o', 'vision_detail': 'low',
}


class FakeEncoding:
    def encode(self, text):
        return text.split()


@pytest.fixture(autouse=True)
def offline_tiktoken(monkeypatch):
    """tiktoken downloads encodings on first use; keep the tests offline and record the fallback."""
    requested = []

    def encoding_for_model(model):
        if model in ('gpt-3.5-turbo', 'gpt-4', 'gpt-4o', 'gpt-4o-mini'):
            return FakeEncoding()
        raise KeyError(model)

    def get_encoding(name):
        requested.append(name)
        return FakeEncoding()

    monkeypatch.setattr(openai_helper.tiktoken, 'encoding_for_model', encoding_for_model)
    monkeypatch.setattr(openai_helper.tiktoken, 'get_encoding', get_encoding)
    return requested


def make_helper(model):
    return OpenAIHelper(config={**CONFIG, 'model': model}, plugin_manager=None)


@pytest.mark.parametrize('model', OLD_MODELS + NEW_MODELS)
def test_default_max_tokens_is_int(model):
    assert isinstance(default_max_tokens(model), int)
    assert default_max_tokens(model) > 0


def test_default_max_tokens_old_values_unchanged():
    assert default_max_tokens('gpt-3.5-turbo') == 1200
    assert default_max_tokens('gpt-4') == 2400
    assert default_max_tokens('gpt-4o-mini') == 4096
    assert default_max_tokens('gpt-6-luna') == 16384


@pytest.mark.parametrize('model', ['gpt-4o', 'gpt-4o-mini'] + NEW_MODELS)
def test_functions_available(model):
    assert are_functions_available(model)


def test_functions_unavailable_for_old_exceptions():
    assert not are_functions_available('gpt-4-0314')
    assert not are_functions_available('o1')


def test_legacy_params_unchanged():
    params = chat_completion_params('gpt-4o-mini', {**CONFIG, 'temperature': 0.7}, 1200, 1)
    assert params == {'temperature': 0.7, 'n': 1, 'max_tokens': 1200,
                      'presence_penalty': 0.0, 'frequency_penalty': 0.0}


@pytest.mark.parametrize('model', NEW_MODELS + ['o1'])
def test_new_model_params_defaults(model):
    assert not is_legacy_model(model)
    assert chat_completion_params(model, CONFIG, 16384, 1) == {'max_completion_tokens': 16384}


def test_new_model_params_explicit_values_are_sent():
    config = {**CONFIG, 'temperature': 0.5, 'presence_penalty': 0.1, 'frequency_penalty': 0.2}
    assert chat_completion_params('gpt-6-luna', config, 100, 2) == {
        'max_completion_tokens': 100, 'temperature': 0.5, 'presence_penalty': 0.1,
        'frequency_penalty': 0.2, 'n': 2}


@pytest.mark.parametrize('model', OLD_MODELS + NEW_MODELS)
def test_count_tokens_and_max_model_tokens_do_not_crash(model):
    helper = make_helper(model)
    messages, tokens = helper.get_conversation_stats(chat_id=1)
    assert messages == 1
    # system message: 3 overhead + 5 words + 3 reply priming (+1 for the role value)
    assert tokens == 3 + 1 + 5 + 3
    assert helper._OpenAIHelper__max_model_tokens() >= 4096


@pytest.mark.parametrize('model', NEW_MODELS)
def test_unknown_model_falls_back_to_o200k(model, offline_tiktoken):
    make_helper(model).get_conversation_stats(chat_id=1)
    assert offline_tiktoken == ['o200k_base']


def test_count_tokens_with_image_for_new_vision_model():
    from io import BytesIO
    from PIL import Image
    from utils import encode_image

    buf = BytesIO()
    Image.new('RGB', (64, 64)).save(buf, format='PNG')
    buf.seek(0)
    helper = OpenAIHelper(config={**CONFIG, 'model': 'gpt-6-luna', 'vision_model': 'gpt-6-luna'},
                          plugin_manager=None)
    helper.reset_chat_history(1)
    helper.conversations[1].append({'role': 'user', 'content': [
        {'type': 'text', 'text': 'what is this'},
        {'type': 'image_url', 'image_url': {'url': encode_image(buf), 'detail': 'low'}}]})
    assert helper.get_conversation_stats(1)[1] > 85


def test_reasoning_effort_sent_for_current_models_only():
    config = {**CONFIG, 'reasoning_effort': 'none'}
    assert chat_completion_params('gpt-6-luna', config, 100, 1)['reasoning_effort'] == 'none'
    assert 'reasoning_effort' not in chat_completion_params('gpt-4o-mini', config, 100, 1)
    assert 'reasoning_effort' not in chat_completion_params('o1', config, 100, 1)
    assert 'reasoning_effort' not in chat_completion_params('gpt-6-luna', {**CONFIG, 'reasoning_effort': ''}, 100, 1)
