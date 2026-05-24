.PHONY: eval test clean

eval:
	python3 eval.py

test:
	python3 -m unittest discover -s tests -v

clean:
	rm -f predictions.jsonl metrics.json
