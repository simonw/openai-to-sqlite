from click.testing import CliRunner
from openai_to_sqlite.cli import cli, encode
import os
import pytest
import sqlite_utils
from unittest import mock

EXAMPLE_CSV = """id,name,description
1,cli,Command line interface
2,sql,Structured query language"""

EXAMPLE_TSV = """id\tname\tdescription
1\tcli\tCommand line interface
2\tsql\tStructured query language"""

EXAMPLE_JSON = """[
    {
        "id": 1,
        "name": "cli",
        "description": "Command line interface"
    },
    {
        "id": 2,
        "name": "sql",
        "description": "Structured query language"
    }
]"""


MOCK_RESPONSE = {
    "usage": {"total_tokens": 3},
    "data": [
        {"index": 0, "embedding": [1.5] * 1536},
        {"index": 1, "embedding": [1.5] * 1536},
    ],
}
MOCK_RESPONSE_BATCH_SIZE_1 = {
    "usage": {"total_tokens": 3},
    "data": [
        {"index": 0, "embedding": [1.5] * 1536},
    ],
}

MOCK_EMBEDDING = encode([1.5] * 1536)


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""})
def test_error_if_no_api_key():
    runner = CliRunner()
    result = runner.invoke(cli, ["embeddings", "test.db", "-"], input="[]")
    assert result.exit_code == 1
    assert "OpenAI API token is required" in result.output


def test_error_if_no_sql_and_no_input_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["embeddings", "test.db"])
    assert result.exit_code == 2
    assert "Error: Either --sql or input path is required" in result.output


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "abc"})
@pytest.mark.parametrize("use_token_option", (True, False))
@pytest.mark.parametrize("format", ("csv", "tsv", "json"))
@pytest.mark.parametrize("use_stdin", (True, False))
@pytest.mark.parametrize("use_explicit_format", (True, False))
@pytest.mark.parametrize("batch_size", (None, 1))
def test_embeddings(
    httpx_mock,
    tmpdir,
    use_token_option,
    format,
    use_stdin,
    use_explicit_format,
    batch_size,
):
    httpx_mock.add_response(
        json=MOCK_RESPONSE if batch_size is None else MOCK_RESPONSE_BATCH_SIZE_1
    )
    db_path = str(tmpdir / "embeddings.db")
    runner = CliRunner()
    args = ["embeddings", db_path]
    data = ""
    if format == "csv":
        data = EXAMPLE_CSV
    elif format == "tsv":
        data = EXAMPLE_TSV
    elif format == "json":
        data = EXAMPLE_JSON
    if use_stdin:
        input = data
        args.append("-")
    else:
        input = None
        input_path = str(tmpdir / "input." + format)
        with open(input_path, "w") as fp:
            fp.write(data)
        args.append(input_path)
    if use_explicit_format:
        args.extend(["--format", format])
    if batch_size:
        args.extend(["--batch-size", str(batch_size)])
    expected_token = "abc"
    if use_token_option:
        args.extend(["--token", "def"])
        expected_token = "def"
    result = runner.invoke(cli, args, input=input)
    assert result.exit_code == 0
    db = sqlite_utils.Database(db_path)
    assert list(db["embeddings"].rows) == [
        {"id": "1", "embedding": MOCK_EMBEDDING},
        {"id": "2", "embedding": MOCK_EMBEDDING},
    ]
    requests = httpx_mock.get_requests()
    assert len(requests) == 1 if batch_size is None else 2
    assert all(r.url == "https://api.openai.com/v1/embeddings" for r in requests)
    assert all(
        r.headers["authorization"] == "Bearer {}".format(expected_token)
        for r in requests
    )


def test_invalid_json_explicit_format():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["embeddings", "test.db", "-", "--format", "json", "--token", "abc"],
        input="Bad JSON",
    )
    assert result.exit_code == 1
    assert result.output == "Error: Expecting value: line 1 column 1 (char 0)\n"


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "abc"})
@pytest.mark.parametrize("use_other_db", (True, False))
@pytest.mark.parametrize("table_option", (None, "-t", "--table"))
def test_sql(httpx_mock, tmpdir, use_other_db, table_option):
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

    expected_table = "embeddings"
    if table_option:
        extra_opts.extend([table_option, "other_table"])
        expected_table = "other_table"

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
    assert list(embeddings_db[expected_table].rows) == [
        {"id": "1", "embedding": MOCK_EMBEDDING},
        {"id": "2", "embedding": MOCK_EMBEDDING},
    ]


@pytest.mark.parametrize("table_option", (None, "-t", "--table"))
def test_search(httpx_mock, tmpdir, table_option):
    httpx_mock.add_response(json=MOCK_RESPONSE)
    db_path = str(tmpdir / "embeddings.db")
    db = sqlite_utils.Database(db_path)
    table = "embeddings"
    if table_option:
        table = "other_table"
    db[table].insert_all(
        [
            {"id": 1, "embedding": MOCK_EMBEDDING},
            {"id": 2, "embedding": MOCK_EMBEDDING},
        ],
        pk="id",
    )
    extra_opts = []
    if table_option:
        extra_opts.extend([table_option, "other_table"])
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "search",
            db_path,
            "test search",
            "--token",
            "abc",
        ]
        + extra_opts,
    )
    assert result.exit_code == 0
    assert result.output == "1.000 1\n1.000 2\n"
