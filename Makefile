# Makefile

install:
	pip install -r requirements.txt

run:
	python final_script/v3/main.py

dash:
	streamlit run final_script/v3/dashboard.py