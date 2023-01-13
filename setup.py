from setuptools import setup
import os

VERSION = "0.2"


def get_long_description():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf8",
    ) as fp:
        return fp.read()


setup(
    name="openai-to-sqlite",
    description="Save OpenAI API results to a SQLite database",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Simon Willison",
    url="https://github.com/simonw/openai-to-sqlite",
    project_urls={
        "Issues": "https://github.com/simonw/openai-to-sqlite/issues",
        "CI": "https://github.com/simonw/openai-to-sqlite/actions",
        "Changelog": "https://github.com/simonw/openai-to-sqlite/releases",
    },
    license="Apache License, Version 2.0",
    version=VERSION,
    packages=["openai_to_sqlite"],
    entry_points="""
        [console_scripts]
        openai-to-sqlite=openai_to_sqlite.cli:cli
    """,
    install_requires=["click", "httpx", "sqlite-utils>=3.28"],
    extras_require={"test": ["pytest", "pytest-httpx"]},
    python_requires=">=3.7",
)
