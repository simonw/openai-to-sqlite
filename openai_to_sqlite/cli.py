import click
import httpx
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
@click.option(
    "-i",
    "--input",
    type=click.File("rb"),
    default="-",
)
@click.option(
    "--token",
    help="OpenAI API key",
    envvar="OPENAI_API_KEY",
)
@click.option(
    "table_name",
    "--table",
    default="embeddings",
    help="Name of the table to store embeddings in",
)
@click.option(
    "as_csv",
    "--csv",
    is_flag=True,
    help="Treat input as CSV",
)
def embeddings(db_path, input, token, table_name, as_csv):
    """
    Store embeddings for one or more text documents

    Input can be CSV, TSV or a JSON list of objects.

    The first column is treated as an ID - all other columns
    are assumed to be text that should be concatenated together
    in order to calculate the embeddings.
    """
    if not token:
        raise click.ClickException(
            "OpenAI API token is required, use --token=x or set the "
            "OPENAI_API_KEY environment variable"
        )
    db = sqlite_utils.Database(db_path)
    table = db[table_name]
    if not table.exists():
        table.create(
            {"id": str, "embedding": bytes},
            pk="id",
        )
    if as_csv:
        rows, _ = rows_from_file(input, Format.CSV)
    else:
        # Auto-detect
        rows, _ = rows_from_file(input)
    # Use a click progressbar
    total_tokens = 0
    skipped = 0
    with click.progressbar(
        rows, label="Fetching embeddings", show_percent=True
    ) as rows:
        for row in rows:
            values = list(row.values())
            id = values[0]
            try:
                table.get(id)
                skipped += 1
                continue
            except sqlite_utils.db.NotFoundError:
                pass
            text = " ".join(values[1:])
            response = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"input": text, "model": "text-embedding-ada-002"},
            )
            if response.status_code == 400:
                click.echo(response.json()["error"], err=True)
                click.echo(f"For ID {id} - skipping", err=True)
                continue
            response.raise_for_status()
            data = response.json()
            total_tokens += data["usage"]["total_tokens"]
            vector = data["data"][0]["embedding"]
            # Encode vector as bytes
            embedding = struct.pack("f" * len(vector), *vector)
            table.insert({"id": id, "embedding": embedding}, replace=True)
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
    "--table",
    default="embeddings",
    help="Name of the table to store embeddings in",
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
    other_vectors = [
        (row["id"], struct.unpack("f" * 1536, row["embedding"])) for row in table.rows
    ]
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
