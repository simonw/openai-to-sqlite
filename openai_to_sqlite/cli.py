import click
import httpx
import json
from sqlite_utils.utils import rows_from_file, Format
import sqlite_utils
import struct


@click.group()
@click.version_option()
def cli():
    "Tool for saving OpenAI API results to a SQLite database"


@cli.command()
@click.argument(
    "db_path",
    type=click.Path(file_okay=True, dir_okay=False, allow_dash=False),
)
@click.argument(
    "input_path",
    type=click.File("rb"),
    required=False,
)
@click.option(
    "--token",
    help="OpenAI API key",
    envvar="OPENAI_API_KEY",
)
@click.option(
    "table_name",
    "-t",
    "--table",
    default="embeddings",
    help="Name of the table to store embeddings in",
)
@click.option(
    "--format",
    type=click.Choice(["json", "csv", "tsv"]),
)
@click.option("--sql", help="Read input using this SQL query")
@click.option(
    "--attach",
    type=(str, click.Path(file_okay=True, dir_okay=False, allow_dash=False)),
    multiple=True,
    help="Additional databases to attach - specify alias and file path",
)
@click.option(
    "--batch-size",
    type=click.IntRange(1, 2048),
    default=100,
    help="Number of rows to send to OpenAI at once. Defaults to 100 - use a higher value if your text is smaller strings.",
)
def embeddings(db_path, input_path, token, table_name, format, sql, attach, batch_size):
    """
    Store embeddings for one or more text documents

    Input can be CSV, TSV or a JSON list of objects.

    The first column is treated as an ID - all other columns
    are assumed to be text that should be concatenated together
    in order to calculate the embeddings.
    """
    if not input_path and not sql:
        raise click.UsageError("Either --sql or input path is required")
    if not token:
        raise click.ClickException(
            "OpenAI API token is required, use --token=x or set the "
            "OPENAI_API_KEY environment variable"
        )
    db = sqlite_utils.Database(db_path)
    for alias, attach_path in attach:
        db.attach(alias, attach_path)
    table = db[table_name]
    if not table.exists():
        table.create(
            {"id": str, "embedding": bytes},
            pk="id",
        )
    expected_length = None
    if sql:
        rows = db.query(sql)
        count_sql = "select count(*) as c from ({})".format(sql)
        expected_length = next(db.query(count_sql))["c"]
    else:
        # Auto-detect
        try:
            rows, _ = rows_from_file(
                input_path, Format[format.upper()] if format else None
            )
        except json.JSONDecodeError as ex:
            raise click.ClickException(str(ex))
    # Use a click progressbar
    total_tokens = 0
    skipped = 0
    with click.progressbar(
        rows, label="Fetching embeddings", show_percent=True, length=expected_length
    ) as rows:
        # Run this batch_size at a time
        for batch in batch_rows(rows, batch_size):
            text_to_send = []
            ids_in_batch = []
            for row in batch:
                values = list(row.values())
                id = values[0]
                try:
                    table.get(id)
                    skipped += 1
                    continue
                except sqlite_utils.db.NotFoundError:
                    pass
                text = " ".join(v or "" for v in values[1:])
                ids_in_batch.append(id)
                text_to_send.append(text)
            # Send to OpenAI, but only if batch is populated - since
            # the skip logic could have resulted in an empty batch
            if text_to_send:
                response = httpx.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={"input": text_to_send, "model": "text-embedding-ada-002"},
                )
                if response.status_code == 400:
                    click.echo(response.json()["error"], err=True)
                    click.echo(f"For IDs: {ids_in_batch} - skipping", err=True)
                    continue
                response.raise_for_status()
                data = response.json()
                total_tokens += data["usage"]["total_tokens"]
                results = data["data"]
                # Each one has an "embedding" and an "index"
                for result in results:
                    embedding = encode(result["embedding"])
                    table.insert(
                        {"id": ids_in_batch[result["index"]], "embedding": embedding},
                        replace=True,
                    )
    click.echo(f"Total tokens used: {total_tokens}", err=True)
    if skipped:
        click.echo(f"Skipped {skipped} rows that already existed", err=True)


@cli.command()
@click.argument(
    "db_path",
    type=click.Path(file_okay=True, dir_okay=False, allow_dash=False),
)
@click.argument("query")
@click.option(
    "--token",
    help="OpenAI API key",
    envvar="OPENAI_API_KEY",
)
@click.option(
    "table_name",
    "-t",
    "--table",
    default="embeddings",
    help="Name of the table containing the embeddings",
)
def search(db_path, query, token, table_name):
    """
    Search embeddings using cosine similarity against a query
    """
    if not token:
        raise click.ClickException(
            "OpenAI API token is required, use --token=x or set the "
            "OPENAI_API_KEY environment variable"
        )
    db = sqlite_utils.Database(db_path)
    table = db[table_name]
    if not table.exists():
        raise click.ClickException(f"Table {table_name} does not exist")
    # Fetch the embedding for the query
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"input": query, "model": "text-embedding-ada-002"},
    )
    response.raise_for_status()
    data = response.json()
    vector = data["data"][0]["embedding"]
    # Now calculate cosine similarity with everything in the database table
    other_vectors = [(row["id"], decode(row["embedding"])) for row in table.rows]
    results = [
        (id, cosine_similarity(vector, other_vector))
        for id, other_vector in other_vectors
    ]
    results.sort(key=lambda r: r[1], reverse=True)
    for id, score in results[:10]:
        print(f"{score:.3f} {id}")


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)


def decode(blob):
    return struct.unpack("f" * 1536, blob)


def encode(values):
    return struct.pack("f" * 1536, *values)


def batch_rows(rows, batch_size):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
