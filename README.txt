You will need install unidecode for running the scripts. You can download and install it
with command 'pip install unidecode'.

Run script: python /src/script.py

Content of repo:

|----- data - folder contains data for building language models in Czech language
	|----- negative.txt - negative posts about films in Czech
	|----- neutral.txt - neutral posts about films in Czech
	|----- positive.txt - positive posts about films in Czech
	|----- licence.txt - license of data
	|----- stopwords.txt - my stopwords which I use in script (stopwords are grouped
						   from several sources)
|----- src - python scripts
	|----- script.py
	|----- czech_stemmer.py - czech stemmer developed by Luís Gomes which I am using
	|----- weight_averaging.py - my script on simple weight averaging results of several
								 models
|----- README