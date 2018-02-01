import re

th = open('test.txt')

th_r = th.read()

sentences = re.split(r'\n\n|(?<![A-Z][a-z]\.)(?<!\s[A-Z]\.)(?<=[.?!])\s+',th_r)

for ptr in sentences:
    ptr = re.sub('([.?!]\')\n', r'\1@',ptr)
    ptr = re.sub('@\n','</s>\n',ptr)
    ptr = re.sub('@','</s> ', ptr)
    print (ptr+'</s>')

