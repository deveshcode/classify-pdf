# Makefile

install:
	pip install -r requirements.txt

run:
	python final_script/v3/run.py

dash:
	streamlit run final_script/v3/dashboard.py

test:
	pytest final_script/v3/tests/test_main.py -v