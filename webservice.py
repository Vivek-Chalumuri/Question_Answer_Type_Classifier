import web
import requests
import os
import time
import process_TREC
import pred_code2
import gensim
import sys
import re
from subprocess import call

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

urls = (
    '/', 'Answer_Type',
    )

model = gensim.models.KeyedVectors.load_word2vec_format("/home/ubuntu/vivek/DCNN/data/google_w2v.bin", binary=True)

class Answer_Type:
	def GET(self):
		text =str(web.input())
		text = text.strip(" <Storage {'")
		text = text.strip("': u''}>")
		print text
        #from getpages import *
		t0 = time.time()
		f = open('TREC/newQuery.txt', 'w')
		f.write(text)  # python will convert \n to os.linesep
		f.close()

		return_code = call("/home/ubuntu/vivek/DCNN/stanford-parser-full-2015-04-20/lexparser.sh TREC/newQuery.txt > log.txt", shell=True)

		f = open("log.txt")
		lines = f.readlines()
		f.close()

		parsed_text = "["

		for line in lines[:-2]:
		    parsed_text += line
	    	parsed_text += ", "

		parsed_text += lines[-2] 
		parsed_text =  re.sub("\n", "", parsed_text) + "]"

		f = open('TREC/newQuery_parsed.txt', 'w')
		f.write(parsed_text)  # python will convert \n to os.linesep
		f.close()

		process_TREC.main(model)
		prediction = pred_code2.main()

		t1 = time.time()
		total_time = t1-t0

		return prediction, total_time

class MyWebApplication(web.application):
	def run(self, port=80, *middleware):
		func = self.wsgifunc(*middleware)
		return web.httpserver.runsimple(func, ('0.0.0.0', port))

if __name__ == "__main__":
	app = MyWebApplication(urls, globals())
	app.run(port=59812)
