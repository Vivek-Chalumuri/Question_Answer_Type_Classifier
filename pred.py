import sys
import re
from subprocess import call
import time
import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
#t0 = time.time()
#text = sys.argv[1]
def Answer_type_prediction(text):
	t0 = time.time()
	f = open('TREC/newQuery.txt', 'w')
	f.write(text)  # python will convert \n to os.linesep
	f.close()

	return_code = call("/home/ubuntu/vivek/DCNN/stanford-parser-full-2015-04-20/lexparser.sh TREC/newQuery.txt > log.txt", shell=True)

	f = open("log.txt")

	lines = f.readlines()

	parsed_text = "["

	for line in lines[:-2]:
	    	parsed_text += line
    		parsed_text += ", "

	parsed_text += lines[-2] 
	parsed_text =  re.sub("\n", "", parsed_text) + "]"

	f = open('TREC/newQuery_parsed.txt', 'w')
	f.write(parsed_text)  # python will convert \n to os.linesep
	f.close()
	import process_TREC
	process_TREC.main()
	import pred_code
	prediction = pred_code.main()
	t1 = time.time()
	print (t1-t0)
	return prediction

