
all: feynr_docs.pdf

feynr_docs.pdf: feynr_docs.tex feynr.sty feynr_render.py Makefile
	python feynr_render.py $<
	pdflatex $<

upload: feynr_docs.pdf Makefile
	scp $< snp@builder.mit.edu:/var/www/data

.PHONY: clean
clean:
	rm -f feynr_docs.pdf

