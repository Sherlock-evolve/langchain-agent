import os

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


@pytest.mark.real_model
def test_real_model_stream_contract():
    if os.getenv("RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("real model integration is opt-in")
    required = (
        "ZHIPU_MODEL",
        "ZHIPU_API_KEY",
        "ZHIPU_BASE_URL",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.fail(
            "missing real model settings: " + ", ".join(missing)
        )

    model = ChatOpenAI(
        model=os.environ["ZHIPU_MODEL"],
        api_key=os.environ["ZHIPU_API_KEY"],
        base_url=os.environ["ZHIPU_BASE_URL"],
        temperature=0,
        max_tokens=32,
        timeout=30,
        max_retries=1,
    )
    chunks = list(
        model.stream(
            [HumanMessage(content="只回复 OK")]
        )
    )
    assert chunks
    assert any(chunk.text.strip() for chunk in chunks)
