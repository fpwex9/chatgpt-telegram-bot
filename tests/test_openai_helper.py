import os
import sys
import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

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


TOOL_SPEC = {'name': 'lookup', 'description': 'Look up a value',
             'parameters': {'type': 'object', 'properties': {'value': {'type': 'string'}}}}
TOOL_CALL = {'id': 'call_1', 'type': 'function',
             'function': {'name': 'lookup', 'arguments': '{"value":"test"}'}}


def completion(tool_calls=None, content='Done'):
    return ChatCompletion.model_validate({
        'id': 'test', 'created': 0, 'model': 'test', 'object': 'chat.completion',
        'choices': [{'index': 0, 'finish_reason': 'tool_calls' if tool_calls else 'stop',
                     'message': {'role': 'assistant', 'content': None if tool_calls else content,
                                 'tool_calls': tool_calls}}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12},
    })


def chunk(delta=None, finish_reason=None, empty=False):
    return ChatCompletionChunk.model_validate({
        'id': 'test', 'created': 0, 'model': 'test', 'object': 'chat.completion.chunk',
        'choices': [] if empty else [{'index': 0, 'delta': delta or {}, 'finish_reason': finish_reason}],
    })


async def stream_items(items):
    for item in items:
        yield item


def text_stream():
    return stream_items([chunk({'role': 'assistant', 'content': ''}),
                         chunk({'content': 'Do'}), chunk({'content': 'ne'}), chunk(finish_reason='stop')])


def tool_response(stream, multiple=False):
    ignored = {**TOOL_CALL, 'id': 'call_2'}
    if not stream:
        return completion([TOOL_CALL, ignored] if multiple else [TOOL_CALL])
    deltas = [{'index': 0, 'id': 'call_', 'type': 'function',
               'function': {'name': 'look', 'arguments': '{"value":'}}]
    if multiple:
        deltas.insert(0, {'index': 1, **ignored})
    return stream_items([
        chunk(empty=True), chunk({'role': 'assistant', 'content': None}),
        chunk({'tool_calls': deltas}), chunk(empty=True),
        chunk({'tool_calls': [{'index': 0, 'id': '1', 'function': {'name': 'up', 'arguments': '"test"'}}]}),
        chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '}'}}]}, finish_reason='tool_calls'),
    ])


@pytest.fixture
def tool_helper(monkeypatch):
    def make(model, responses, **config):
        requests = []
        responses = iter(responses)

        async def create(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=create))))
        monkeypatch.setattr(openai_helper.openai, 'AsyncOpenAI', Mock(return_value=client))
        plugin_manager = Mock()
        plugin_manager.get_functions_specs.return_value = [TOOL_SPEC]
        plugin_manager.call_function = AsyncMock(return_value='{"result": "found"}')
        helper = OpenAIHelper(config={
            **CONFIG, 'model': model, 'bot_language': 'en', 'enable_functions': True,
            'max_tokens': 100, 'max_history_size': 100, 'max_conversation_age_minutes': 60,
            'n_choices': 1, 'functions_max_consecutive_calls': 10, 'reasoning_effort': 'none',
            'show_usage': False, 'show_plugins_used': False, 'vision_max_tokens': 100,
            'enable_vision_follow_up_questions': True, **config,
        }, plugin_manager=plugin_manager)
        return helper, requests

    return make


async def chat_response(helper, stream, query='Find it'):
    if stream:
        return [item async for item in helper.get_chat_response_stream(1, query)][-1]
    return await helper.get_chat_response(1, query)


def assert_tool_history(messages):
    pending = set()
    for message in messages:
        assert message['role'] != 'function'
        if message['role'] == 'tool':
            assert message['tool_call_id'] in pending
            pending.remove(message['tool_call_id'])
        else:
            assert not pending
            pending = {call['id'] for call in message.get('tool_calls', [])}
    assert not pending


@pytest.mark.parametrize('model', ['gpt-4o-mini', 'gpt-6-luna'])
@pytest.mark.parametrize('stream', [False, True])
def test_tool_calls_and_follow_up_history(tool_helper, model, stream):
    helper, requests = tool_helper(model, [tool_response(stream, multiple=True),
                                          text_stream() if stream else completion()])
    answer, tokens = asyncio.run(chat_response(helper, stream))
    assert answer == 'Done'
    assert int(tokens) > 0
    helper.plugin_manager.call_function.assert_awaited_once_with('lookup', helper, TOOL_CALL['function']['arguments'])
    assert len(requests) == 2
    for request in requests:
        assert request['tools'] == [{'type': 'function', 'function': TOOL_SPEC}]
        assert request['tool_choice'] == 'auto'
        assert request['stream'] == stream
        assert 'functions' not in request and 'function_call' not in request
        assert request.get('reasoning_effort') == ('none' if model == 'gpt-6-luna' else None)
        assert_tool_history(request['messages'])
    assert requests[1]['messages'][-2:] == [
        {'role': 'assistant', 'content': None, 'tool_calls': [TOOL_CALL]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"result": "found"}'},
    ]
    assert_tool_history(helper.conversations[1])
    assert helper.get_conversation_stats(1)[1] > 0


@pytest.mark.parametrize('stream', [False, True])
def test_direct_tool_result_has_complete_history(tool_helper, stream):
    helper, requests = tool_helper('gpt-6-luna', [tool_response(stream, multiple=True),
                                                text_stream() if stream else completion()])
    direct = json.dumps({'direct_result': True, 'kind': 'photo', 'value': 'test'})
    helper.plugin_manager.call_function.return_value = direct
    assert asyncio.run(chat_response(helper, stream)) == (direct, '0')
    assert len(requests) == 1
    assert helper.conversations[1][-2] == {'role': 'assistant', 'content': None, 'tool_calls': [TOOL_CALL]}
    assert helper.conversations[1][-1]['role'] == 'tool'
    assert 'Done' in helper.conversations[1][-1]['content']
    assert_tool_history(helper.conversations[1])
    assert asyncio.run(chat_response(helper, stream, 'Next'))[0] == 'Done'
    assert_tool_history(requests[1]['messages'])


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('limit', [0, 1])
def test_tool_choice_none_at_consecutive_call_limit(tool_helper, stream, limit):
    responses = [tool_response(stream) for _ in range(limit + 1)]
    responses.append(text_stream() if stream else completion())
    helper, requests = tool_helper('gpt-6-luna', responses, functions_max_consecutive_calls=limit)
    assert asyncio.run(chat_response(helper, stream))[0] == 'Done'
    assert [request['tool_choice'] for request in requests] == ['auto'] * (limit + 1) + ['none']
    for request in requests:
        assert_tool_history(request['messages'])


def test_plain_stream_keeps_first_content_chunk(tool_helper):
    helper, requests = tool_helper('gpt-6-luna', [stream_items([
        chunk(empty=True), chunk({'content': 'Do'}), chunk({'content': 'ne'}), chunk(finish_reason='stop'),
    ])])
    assert asyncio.run(chat_response(helper, True))[0] == 'Done'
    helper.plugin_manager.call_function.assert_not_awaited()
    assert len(requests) == 1


@pytest.mark.parametrize('summary_fails', [False, True])
@pytest.mark.parametrize('vision', [False, True])
@pytest.mark.parametrize('history_size', [3, 4])
def test_summary_and_trimming_keep_tool_exchanges_valid(tool_helper, summary_fails, vision, history_size):
    helper, requests = tool_helper('gpt-6-luna', [
        RuntimeError('Summary unavailable') if summary_fails else completion(content='Summary'), completion(),
    ], max_history_size=history_size)
    helper.reset_chat_history(1)
    helper.conversations[1].extend([
        {'role': 'user', 'content': 'Old question'},
        {'role': 'assistant', 'content': None, 'tool_calls': [TOOL_CALL]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'Old result'},
        {'role': 'assistant', 'content': 'Old answer'},
    ])
    if vision:
        asyncio.run(helper._OpenAIHelper__common_get_chat_response_vision(1, [{'type': 'text', 'text': 'Next'}]))
    else:
        asyncio.run(helper.get_chat_response(1, 'Next'))
    assert len(requests) == 2
    # Summarisation embeds history as text; it never sends standalone tool messages.
    assert [message['role'] for message in requests[0]['messages']] == ['assistant', 'user']
    assert 'call_1' in requests[0]['messages'][1]['content']
    assert_tool_history(requests[1]['messages'])
    if summary_fails:
        assert len(requests[1]['messages']) <= history_size
        assert any(message.get('tool_calls') for message in requests[1]['messages']) == (history_size == 4)
    else:
        assert requests[1]['messages'][1] == {'role': 'assistant', 'content': 'Summary'}
