from llmify import BaseChatModel


class DummyModel(BaseChatModel):
    async def invoke(self, messages, **kwargs):
        return self._merge_params(kwargs)

    async def invoke_structured(self, messages, response_model, **kwargs):
        pass

    async def stream(self, messages, **kwargs):
        pass


def test_default_params():
    model = DummyModel(max_tokens=100, temperature=0.7)
    params = model._merge_params({})

    assert params["max_tokens"] == 100
    assert params["temperature"] == 0.7


def test_override_params():
    model = DummyModel(max_tokens=100, temperature=0.7)
    params = model._merge_params({"max_tokens": 200})

    assert params["max_tokens"] == 200
    assert params["temperature"] == 0.7


def test_none_params_filtered():
    model = DummyModel(max_tokens=None)
    params = model._merge_params({})

    assert "max_tokens" not in params
