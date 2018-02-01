import re
import numpy as np
from sklearn import tree
from nltk.tokenize import sent_tokenize


def extract_punct(text):
    data = []
    whitespace = [" ","\t","\n","\r","\f","\v"]
    for m in re.finditer("[^\s\w</s>]", text):
        #print (m.end(),len(text))
        if m.start() == 0 or m.end()>=len(text)-10:
            continue
        else:
            curr = m.start()
            prev = curr-1
            prev_word = text[prev]
            k=0
            while text[prev] not in whitespace:
                k=k+1
                prev = prev - 1
                prev_word = text[prev]+prev_word
            if k!=0:
                prev_word = prev_word[1:]

            next = m.end()
            if text[next:next+4] == "</s>":
                terminator = 1                  # label 1 if punctuation is sentence terminator otherwise label 0
                next = next+4
            elif text[next+1:next+5] == "</s>":
                terminator = 1
            else:
                terminator = 0
            while text[next] in whitespace:
                next=next+1
            next_word = text[next]
            l=0
            while text[next] not in whitespace:
                l=l+1
                next = next+1
                next_word = next_word+text[next]
            if l!=0:
                next_word = next_word[:-1]
            data.append([prev_word,m.group(0),next_word,terminator])
            #print([prev_word,m.group(0),next_word,terminator])
    punct_list = list(set([item[1] for item in data]))
    return data,punct_list


def make_feature(data,punct_list):
    output = []
    feature_vectors = []
    prev_capital = []
    next_capital = []
    prev_word_is_short = []  # to check if prev word length is less than 3(abbreviation)
    next_quote = []  # to check if Current is “.”,”?”, or “!” and Next is double left quote (“)
    curr_quote = []  # Previous is “.”, ”?”, or “!”, Current is (' or ”) and Next is (“) or is uppercase
    closing_quote = []

    for index, value in enumerate(data):
        prev_word = value[0]
        curr_word = value[1]
        next_word = value[2]
        output.append(value[3])

        prev_capital.append(1 if prev_word[0].isupper() else 0)
        next_capital.append(1 if next_word[0].isupper() else 0)
        prev_word_is_short.append(1 if len(prev_word) < 3 else 0)
        next_quote.append(1 if curr_word in [".","?","!"] and next_word[0] in ["\"","\'"] else 0)
        curr_quote.append(1 if prev_word in [".", "?", "!"] and curr_word[0] in ["\"", "\'"] and next_word[0]=="\"" else 0)
        closing_quote.append(1 if prev_word[-1] in [".", "?", "!"] and curr_word[0] in ["\"", "\'"] and next_word[0].isupper() else 0)

        one_hot_word_vector = [0] * len(punct_list)
        pos = punct_list.index(curr_word)
        one_hot_word_vector[pos] = 1
        #print("ohwv", one_hot_word_vector)
        feature_vector = one_hot_word_vector
        feature_vector.extend([prev_capital[index],next_capital[index],prev_word_is_short[index],next_quote[index],curr_quote[index],closing_quote[index]])

        #print("fvec", feature_vector)
        feature_vectors.append(feature_vector)

    return (feature_vectors,output)


def main():
    f = open('test.txt', 'r')
    text = f.read()
    f.close()

    sent_tokenize_list = sent_tokenize(text)
    for index, item in enumerate(sent_tokenize_list):
        item = item + "</s>"
        sent_tokenize_list[index] = item
    text = "\n".join(sent_tokenize_list)


    data,punct_list = extract_punct(text)
    training_set = data[0:int(0.7*len(data))]
    test_set     = data[int(0.7*len(data)):]
    training_vectors, training_output = make_feature(training_set,punct_list)
    #print(training_vectors)
    test_vectors, test_output     = make_feature(test_set,punct_list)
    #print(test_vectors)

    classifier = tree.DecisionTreeClassifier()
    print('Training our decision tree...')
    classifier.fit(training_vectors, training_output)  # 1 = EOS, 0 = NEOS
    print('Training complete!')

    total_seen = 0
    total_correct = 0

    for i, test_example in enumerate(test_vectors):
        #print(test_example)
        correct = test_output[i]
        #print(test_example, correct)
        #print(np.array(test_example).reshape(1, -1))
        pred = classifier.predict(np.array(test_example).reshape(1,-1))

        print(i,test_set[i], 'Predicted:', pred[0], 'Actual:', correct)

        if str(pred[0]) == str(correct):
            total_correct += 1
            total_seen += 1
        else:
            total_seen += 1

        accuracy = (total_correct/total_seen)*100
    print('Model Accuracy:', accuracy)

main()