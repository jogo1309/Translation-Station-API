import string
#get data line by line and process it
def pre_process_data(path, is_output):
    unique_sentances = []
    unique_words = set()
    f = open(path, encoding="utf8")
    for line in f.readlines():
        #convert to lower case, strip punctuation, remove newlines and add to sentance array
        line = line.lower().translate(str.maketrans('', '', string.punctuation)).replace("\n", "")
        if(is_output):
            #add start and end tokens to translations
            line = "[[" + line + "]]"
        unique_sentances.append(line)
        print(line)
        #add to unique words found
        words = line.split(" ")
        for word in words:
            #print(word)
            if word not in unique_words:
                unique_words.add(word)
    
    return unique_sentances, sorted(list(unique_words)), len(unique_words), max([len(txt.split(' ')) for txt in unique_sentances])