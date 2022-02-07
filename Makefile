
include .env

TAG:=$(shell git rev-list HEAD --max-count=1 --abbrev-commit)

install-pyenv-version:
	@pyenv install

install-poetry:
	@python -m pip install --upgrade pip
	@python -m pip install poetry

install-dependencies:
	@python -m poetry install
	@python -m pip install --upgrade pre-commit
	@python -m poetry run pre-commit install --hook-type commit-msg

init: install-pyenv-version install-poetry install-dependencies

all:
	@python -m poetry run pre-commit run --all-files
