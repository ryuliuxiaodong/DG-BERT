from nltk.tokenize import word_tokenize
import pandas as pd
import numpy



def recursively_find_refined_tokens(word, word_2_definition, key_features, refined_definition_tokens, already_searched_words):
    if word in already_searched_words:
        return
    already_searched_words.append(word)
    try:
        definition = word_2_definition[word]
        definition_tokens = word_tokenize(definition)
        for definition_token in definition_tokens:
            if definition_token in key_features:
                if definition_token not in refined_definition_tokens:
                    refined_definition_tokens.append(definition_token)
            else:
                if definition_token in already_searched_words:
                    continue
                else:
                    recursively_find_refined_tokens(definition_token, word_2_definition, key_features, refined_definition_tokens, already_searched_words)
        return
    except:
        return


if __name__ == "__main__":

    word2definitions_file = open(r'./entry_2_filtered_definition_example.csv', 'r', encoding='utf-16')
    word2definitions_list = word2definitions_file.readlines()
    word2definitions_file.close()

    word_2_definition = {}

    for record in word2definitions_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')

        word_2_definition[all_values[0]] = all_values[1]

    feature_frequency_file = open(r'./feature_frequency.csv', 'r', encoding='utf-16')
    feature_frequency_list = feature_frequency_file.readlines()
    feature_frequency_file.close()

    features = []

    for record in feature_frequency_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')

        features.append(all_values[0])

    key_features = features[0:10748]

    whole_list = []
    count = 0

    for record in word2definitions_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')

        tmp = []
        tmp.append(all_values[0])

        definition_tokens = word_tokenize(all_values[1])

        refined_definition_tokens = []
        already_searched_words = []

        for definition_token in definition_tokens:
            if definition_token in key_features:
                if definition_token not in refined_definition_tokens:
                    refined_definition_tokens.append(definition_token)
            else:
                recursively_find_refined_tokens(definition_token, word_2_definition, key_features, refined_definition_tokens, already_searched_words)

        if len(refined_definition_tokens) == 0:
            tmp.append(all_values[1])
        else:
            tmp.append(' '.join([elem.lower() for elem in refined_definition_tokens]))
        whole_list.append(tmp)

        count += 1
        print(count)


    matrix = numpy.matrix(whole_list)
    dataframe = pd.DataFrame(data=matrix.astype(str))
    dataframe.to_csv(r'./entry_2_extracted_features_example.csv', sep='\t', header=False, index=False, encoding='utf-16')

