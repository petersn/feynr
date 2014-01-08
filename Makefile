
all: feynr_docs.pdf

feynr_docs.pdf: feynr_docs.tex feynr.sty feynr_render.py Makefile
	python feynr_render.py $<
	pdflatex $<

.PHONY: clean
clean:
	rm -f feynr_docs.pdf

