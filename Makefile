.PHONY: setup reset test run run-auto bank eval eval-scripted compare dashboard

setup:
	pip install -r requirements.txt

reset:
	python -m sandbox.reset

test:
	python -m pytest -q

run:
	python -m agent.run --reset

run-auto:
	python -m agent.run --reset --approver auto

bank:
	uvicorn sandbox.bank_api:app --port 8001

eval:
	python -m eval.run_eval --model real

eval-scripted:
	python -m eval.run_eval --model scripted

compare:
	python -m eval.compare --model scripted

dashboard:
	python -m agent.dashboard --db sandbox/sandbox.db
