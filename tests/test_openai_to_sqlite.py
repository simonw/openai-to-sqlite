from click.testing import CliRunner
from openai_to_sqlite.cli import cli
import sqlite_utils

EXAMPLE_CSV = """id,name,description
1,cli,Command line interface
2,sql,Structured query language"""


def test_csv(httpx_mock, tmpdir):
    httpx_mock.add_response(
        json={
            "usage": {"total_tokens": 3},
            "data": [{"embedding": [1, 2, 3]}],
        }
    )
    db_path = str(tmpdir / "embeddings.db")
    runner = CliRunner()
    result = runner.invoke(cli, ["embeddings", db_path], input=EXAMPLE_CSV)
    assert result.exit_code == 0
    db = sqlite_utils.Database(db_path)
    assert list(db["embeddings"].rows) == [
        {"id": "1", "embedding": b"\x00\x00\x80?\x00\x00\x00@\x00\x00@@"},
        {"id": "2", "embedding": b"\x00\x00\x80?\x00\x00\x00@\x00\x00@@"},
    ]
