from click.testing import CliRunner
from openai_to_sqlite.cli import cli
import os
import pytest
import sqlite_utils
from unittest import mock

EXAMPLE_CSV = """id,name,description
1,cli,Command line interface
2,sql,Structured query language"""

MOCK_RESPONSE = {
    "usage": {"total_tokens": 3},
    "data": [{"embedding": [1, 2, 3]}],
}
MOCK_EMBEDDING = b"\x00\x00\x80?\x00\x00\x00@\x00\x00@@"


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""})
def test_error_if_no_api_key():
    runner = CliRunner()
    result = runner.invoke(cli, ["embeddings", "test.db"], input="[]")
    assert result.exit_code == 1
    assert "OpenAI API token is required" in result.output


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "abc"})
@pytest.mark.parametrize("use_token_option", (True, False))
def test_csv(httpx_mock, tmpdir, use_token_option):
    httpx_mock.add_response(json=MOCK_RESPONSE)
    db_path = str(tmpdir / "embeddings.db")
    runner = CliRunner()
    args = ["embeddings", db_path]
    expected_token = "abc"
    if use_token_option:
        args.extend(["--token", "def"])
        expected_token = "def"
    result = runner.invoke(cli, args, input=EXAMPLE_CSV)
    assert result.exit_code == 0
    db = sqlite_utils.Database(db_path)
    assert list(db["embeddings"].rows) == [
        {"id": "1", "embedding": MOCK_EMBEDDING},
        {"id": "2", "embedding": MOCK_EMBEDDING},
    ]
    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    assert all(r.url == "https://api.openai.com/v1/embeddings" for r in requests)
    assert all(
        r.headers["authorization"] == "Bearer {}".format(expected_token)
        for r in requests
    )


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "abc"})
@pytest.mark.parametrize("use_other_db", (True, False))
@pytest.mark.parametrize("table", (None, "custom_embeddings"))
def test_sql(httpx_mock, tmpdir, use_other_db, table):
    httpx_mock.add_response(json=MOCK_RESPONSE)
    db_path = str(tmpdir / "embeddings.db")
    db = sqlite_utils.Database(db_path)
    extra_opts = []
    other_table = "content"
    if use_other_db:
        db_path2 = str(tmpdir / "other.db")
        db = sqlite_utils.Database(db_path2)
        extra_opts = ["--attach", "other", db_path2]
        other_table = "other.content"

    if table:
        extra_opts.extend(["--table", table])

    db["content"].insert_all(
        [
            {"id": 1, "name": "cli", "description": "Command line interface"},
            {"id": 2, "name": "sql", "description": "Structured query language"},
        ],
        pk="id",
    )
    runner = CliRunner()
    args = ["embeddings", db_path, "--sql", "select * from {}".format(other_table)]
    args.extend(extra_opts)
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    embeddings_db = sqlite_utils.Database(db_path)
    assert list(embeddings_db[table or "embeddings"].rows) == [
        {"id": "1", "embedding": MOCK_EMBEDDING},
        {"id": "2", "embedding": MOCK_EMBEDDING},
    ]
