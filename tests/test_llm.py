import json

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

import aimedicine.llm as llm


FAKE_KEY = "test-only-never-a-real-api-key"
PRIVATE_PROVIDER_TEXT = "provider-private-error-body-do-not-echo"


class StructuredReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


def completion(answer="verified", *, usage=None, finish_reason="stop", model="mock-small"):
    return {
        "model": model,
        "choices": [{"finish_reason": finish_reason, "message": {"content": json.dumps({"answer": answer})}}],
        "usage": usage or {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
    }


@pytest.fixture
def client_factory(monkeypatch, tmp_path):
    
    monkeypatch.setattr(llm, "load_dotenv", lambda *args, **kwargs: None)
    for key in ("MISTRAL_API_KEY", "MISTRAL_MODEL", "MISTRAL_BASE_URL", "MISTRAL_TEMPERATURE",
                "MISTRAL_MAX_TOKENS", "MISTRAL_MAX_CALLS", "MISTRAL_MIN_INTERVAL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MISTRAL_API_KEY", FAKE_KEY)
    monkeypatch.setenv("MISTRAL_MODEL", "mock-small")
    monkeypatch.setenv("MISTRAL_MIN_INTERVAL", "0")
    monkeypatch.setattr(llm.time, "sleep", lambda delay: None)
    clients = []

    def create(handler, max_calls=4):
        client = llm.MistralClient(cache_dir=tmp_path / "cache", max_calls=max_calls,
                                  transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    yield create
    for client in clients:
        client.close()


def call(client, payload=None):
    return client.complete("planner", "Return the requested structured reply.",
                           payload or {"question": "synthetic example"}, StructuredReply)


def test_zero_model_allowance_stops_without_repeated_requests(client_factory):
    client = client_factory(lambda request: httpx.Response(429, headers={"x-ratelimit-limit-req-minute": "0"}, json={"message": "Rate limit exceeded"}))
    with pytest.raises(llm.LLMError, match="zero request allowance"):
        call(client)
    assert client.calls == 1


def test_request_uses_official_endpoint_and_json_schema(client_factory):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json=completion())

    client = client_factory(handler)
    assert call(client).answer == "verified"
    request = captured[0]
    assert str(request.url) == "https://api.mistral.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer " + FAKE_KEY
    body = json.loads(request.content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"]["additionalProperties"] is False


def test_success_is_cached_without_second_request_or_billed_usage(client_factory, tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion())

    client = client_factory(handler, max_calls=1)
    assert call(client) == call(client)
    assert len(requests) == client.calls == 1
    summary = client.summary()
    assert summary["cache_hits"] == 1
    assert summary["usage"] == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
    assert [record["cache"] for record in summary["records"]] == [False, True]
    cache_text = next((tmp_path / "cache").glob("*.json")).read_text()
    assert FAKE_KEY not in cache_text
    assert FAKE_KEY not in json.dumps(summary)


def test_different_input_is_not_reused_from_cache(client_factory):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion(answer=f"answer-{len(requests)}"))

    client = client_factory(handler)
    assert call(client, {"question": "one"}).answer == "answer-1"
    assert call(client, {"question": "two"}).answer == "answer-2"
    assert client.calls == 2 and client.cache_hits == 0


def test_second_distinct_request_respects_per_run_budget(client_factory):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion())

    client = client_factory(handler, max_calls=1)
    call(client, {"question": "one"})
    with pytest.raises(llm.LLMError, match="limit reached"):
        call(client, {"question": "two"})
    assert len(requests) == client.calls == 1


def test_retries_also_consume_request_budget(client_factory):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(429, json={"error": PRIVATE_PROVIDER_TEXT})

    client = client_factory(handler, max_calls=2)
    with pytest.raises(llm.LLMError, match="limit reached"):
        call(client)
    assert len(requests) == client.calls == 2
    assert client.summary()["records"] == []


def test_zero_budget_sends_no_request(client_factory):
    def handler(request):
        pytest.fail("A request was sent with zero budget")

    client = client_factory(handler, max_calls=0)
    with pytest.raises(llm.LLMError, match="limit reached"):
        call(client)
    assert client.calls == 0


def test_429_retry_then_success_counts_attempts_and_backoff(client_factory, monkeypatch):
    statuses = [429, 200]
    sleeps = []

    def handler(request):
        status = statuses.pop(0)
        if status == 429:
            return httpx.Response(status, json={"error": PRIVATE_PROVIDER_TEXT}, headers={"retry-after": "9"})
        return httpx.Response(status, json=completion())

    client = client_factory(handler)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    assert call(client).answer == "verified"
    assert client.calls == 2 and sleeps == [9.0]
    assert client.summary()["usage"]["total_tokens"] == 15


@pytest.mark.parametrize("retry_after,expected", [("999", 30.0), ("1", 3.0), ("not-a-number", 5.0)])
def test_retry_after_is_bounded_and_malformed_header_is_safe(client_factory, monkeypatch, retry_after, expected):
    attempts = []
    sleeps = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": retry_after})
        return httpx.Response(200, json=completion())

    client = client_factory(handler)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    assert call(client).answer == "verified"
    assert sleeps == [expected]


def test_retry_limit_is_three_even_if_budget_is_larger(client_factory):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(503, text=FAKE_KEY + PRIVATE_PROVIDER_TEXT)

    client = client_factory(handler, max_calls=10)
    with pytest.raises(llm.LLMError) as error:
        call(client)
    assert len(requests) == client.calls == 3
    assert PRIVATE_PROVIDER_TEXT not in str(error.value)
    assert FAKE_KEY not in str(error.value)


def test_network_errors_retry_boundedly_and_hide_exception_body(client_factory):
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ConnectError(FAKE_KEY + PRIVATE_PROVIDER_TEXT, request=request)

    client = client_factory(handler, max_calls=10)
    with pytest.raises(llm.LLMError, match="bounded retries") as error:
        call(client)
    assert len(requests) == client.calls == 3
    assert PRIVATE_PROVIDER_TEXT not in str(error.value)
    assert FAKE_KEY not in str(error.value)


@pytest.mark.parametrize("status", [400, 401, 403])
def test_nonretryable_errors_hide_provider_body_and_key(client_factory, status):
    client = client_factory(lambda request: httpx.Response(status, text=FAKE_KEY + PRIVATE_PROVIDER_TEXT))
    with pytest.raises(llm.LLMError, match=f"HTTP {status}") as error:
        call(client)
    assert client.calls == 1
    assert PRIVATE_PROVIDER_TEXT not in str(error.value)
    assert FAKE_KEY not in str(error.value)


def test_redirect_is_not_followed_with_credentials(client_factory):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(307, headers={"location": "https://not-mistral.invalid/collect"})

    client = client_factory(handler)
    with pytest.raises(llm.LLMError, match="HTTP 307"):
        call(client)
    assert len(requests) == 1


@pytest.mark.parametrize("base_url", ["http://api.mistral.ai/v1", "https://not-mistral.invalid/v1",
                                      "https://api.mistral.ai.evil.invalid/v1", "https://api.mistral.ai/v2"])
def test_nonofficial_endpoint_is_rejected_before_credentials_can_be_sent(client_factory, monkeypatch, base_url):
    monkeypatch.setenv("MISTRAL_BASE_URL", base_url)
    with pytest.raises(llm.LLMError, match="official HTTPS"):
        client_factory(lambda request: pytest.fail("Credential would be sent to an untrusted endpoint"))


def test_missing_key_fails_before_transport(client_factory, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY")
    client = client_factory(lambda request: pytest.fail("Missing credential must not trigger an HTTP call"))
    with pytest.raises(llm.LLMError, match="MISTRAL_API_KEY is missing"):
        call(client)
    assert client.calls == 0


@pytest.mark.parametrize("content", ["not JSON", '{"answer":', '{"wrong_field":"private"}'])
def test_invalid_model_output_is_not_cached_or_substituted(client_factory, tmp_path, content):
    body = completion()
    body["choices"][0]["message"]["content"] = content + (" " if content == "not JSON" else "")
    client = client_factory(lambda request: httpx.Response(200, json=body))
    with pytest.raises(llm.LLMError, match="invalid structured result"):
        call(client)
    assert list((tmp_path / "cache").glob("*.json")) == []
    assert client.records == []


def test_schema_error_does_not_echo_invalid_secret_like_content(client_factory):
    body = completion()
    body["choices"][0]["message"]["content"] = json.dumps({"wrong_field": FAKE_KEY + PRIVATE_PROVIDER_TEXT})
    client = client_factory(lambda request: httpx.Response(200, json=body))
    with pytest.raises(llm.LLMError) as error:
        call(client)
    assert FAKE_KEY not in str(error.value)
    assert PRIVATE_PROVIDER_TEXT not in str(error.value)


def test_truncated_reply_is_rejected_even_if_partial_content_happens_to_be_valid_json(client_factory, tmp_path):
    client = client_factory(lambda request: httpx.Response(200, json=completion(finish_reason="length")))
    with pytest.raises(llm.LLMError, match="no partial plan accepted"):
        call(client)
    assert client.records == []
    assert list((tmp_path / "cache").glob("*.json")) == []
    
    assert client.summary()["usage"]["total_tokens"] == 15


@pytest.mark.parametrize("body", [{"choices": []}, [], {"choices": [{"message": {"content": ["malformed-block"]}}]}])
def test_malformed_provider_response_shapes_are_normalized_to_safe_errors(client_factory, body):
    client = client_factory(lambda request: httpx.Response(200, json=body))
    with pytest.raises(llm.LLMError, match="invalid structured result"):
        call(client)
    assert client.records == []


def test_non_json_provider_response_is_safe(client_factory):
    client = client_factory(lambda request: httpx.Response(200, text=FAKE_KEY + PRIVATE_PROVIDER_TEXT))
    with pytest.raises(llm.LLMError, match="invalid structured result") as error:
        call(client)
    assert FAKE_KEY not in str(error.value)
    assert PRIVATE_PROVIDER_TEXT not in str(error.value)


def test_text_blocks_response_is_supported(client_factory):
    body = completion()
    body["choices"][0]["message"]["content"] = [
        {"type": "text", "text": '{"answer":'}, {"type": "text", "text": '"verified"}'},
    ]
    client = client_factory(lambda request: httpx.Response(200, json=body))
    assert call(client).answer == "verified"


def test_usage_summary_accumulates_successes_and_reports_actual_model(client_factory):
    client = client_factory(lambda request: httpx.Response(200, json=completion(model="provider-model-version")))
    call(client, {"question": "first"})
    call(client, {"question": "second"})
    summary = client.summary()
    assert summary["api_attempts"] == 2
    assert summary["cache_hits"] == 0
    assert summary["usage"] == {"prompt_tokens": 22, "completion_tokens": 8, "total_tokens": 30}
    assert summary["max_calls"] == 4
    assert len(summary["records"]) == 2
    assert all(record["model"] == "provider-model-version" for record in summary["records"])
    assert FAKE_KEY not in json.dumps(summary)


def test_requested_output_budget_is_capped(client_factory, monkeypatch):
    monkeypatch.setenv("MISTRAL_MAX_TOKENS", "10000")
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    client = client_factory(handler)
    call(client)
    assert bodies[0]["max_tokens"] == client.summary()["max_output_tokens"] == 3200
