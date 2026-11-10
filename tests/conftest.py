import sqlite3

import pytest
from fastapi.testclient import TestClient

from agent.tools import Toolbox
from sandbox.bank_api import create_app
from sandbox.reset import reset


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    reset(path)
    return path


@pytest.fixture
def bank(db_path):
    return TestClient(create_app(db_path))


@pytest.fixture
def toolbox(db_path, bank):
    return Toolbox(db_path, bank)


@pytest.fixture
def query(db_path):
    def q(sql, *params):
        con = sqlite3.connect(db_path)
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()
    return q
