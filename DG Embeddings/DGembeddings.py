import torch
from transformers import BertTokenizer
import numpy




def get_unknown_token_ids(tokenizer, words):
    seen_token_ids = []

    for word in words:
        token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
        for token_id in token_ids:
            if token_id in seen_token_ids:
                continue
            seen_token_ids.append(token_id)

    unseen_token_ids = []

    for index in range(30522):
        if index in seen_token_ids:
            continue
        unseen_token_ids.append(index)

    return unseen_token_ids


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    ### load word2definition tables
    word2definitions_file = open(r'./entry_2_extracted_features_example.csv', 'r', encoding='utf-16')
    word2definitions_list = word2definitions_file.readlines()
    word2definitions_file.close()

    print("word2definitions_list load done")

    word_list = []

    for record in word2definitions_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')
        word_list.append(all_values[0])

    vocab_size = len(word_list)

    unknown_token_ids = get_unknown_token_ids(tokenizer, word_list)
    print("token ids are not contained in the vocabulary like [unused0]: " + str(len(unknown_token_ids)))

    del word2definitions_list
    print("word2definitions_list deleted")
    print(len(word_list))

    ### load latent representations
    matrix_file = open(r'./svd_latent_representation_768d.csv', 'r')
    matrix_table = matrix_file.readlines()
    matrix_file.close()

    print("matrix_table load done")

    latent_representations = []
    count = 0

    for record in matrix_table:
        tmp = []
        record = record.rstrip("\n")
        all_values = record.split('\t')
        for value in all_values:
            tmp.append(float(value))
        latent_representations.append(tmp)
        count += 1

    del matrix_table
    print("matrix_table deleted")
    print(len(latent_representations))

    latent_representations = numpy.array(latent_representations)


    ### load sub_token list from BERT vocabulary
    vocab_file = open(r'./vocab.txt', 'r', encoding='utf-8')
    vocab_table = vocab_file.readlines()
    vocab_file.close()

    sub_tokens = []
    for record in vocab_table:
        record = record.rstrip("\n")
        sub_tokens.append(record)

    print(len(sub_tokens))
    print(sub_tokens.index('[CLS]'))
    print(sub_tokens.index('[SEP]'))
    print(sub_tokens.index('[MASK]'))
    print(sub_tokens.index('[PAD]'))

    DGembeddings = numpy.zeros((30522, 768))

    word_or_phrase_subtokens_list = []
    for word_or_phrase in word_list:
        word_or_phrase_subtokens = tokenizer.tokenize(word_or_phrase)
        word_or_phrase_subtokens_list.append(word_or_phrase_subtokens)
    print(word_or_phrase_subtokens_list[5000])

    for sub_token in sub_tokens:
        print(sub_tokens.index(sub_token))
        print(sub_token)
        if sub_tokens.index(sub_token) in unknown_token_ids:
            continue
        sub_token_occurence_count = 0
        sub_token_embeddings = numpy.zeros(768)
        for word_or_phrase_index in range(vocab_size):
            word_or_phrase_subtokens = word_or_phrase_subtokens_list[word_or_phrase_index]
            if sub_token in word_or_phrase_subtokens:
                sub_token_occurence_count += 1
                word_or_phrase_latent_representation = latent_representations[word_or_phrase_index]
                word_or_phrase_subtokens_length = len(word_or_phrase_subtokens)
                word_or_phrase_latent_representation = word_or_phrase_latent_representation / word_or_phrase_subtokens_length
                sub_token_embeddings += word_or_phrase_latent_representation
        if sub_token_occurence_count == 0:
            continue
        else:
            sub_token_index = sub_tokens.index(sub_token)
            DGembeddings[sub_token_index] = sub_token_embeddings / sub_token_occurence_count

    print(DGembeddings[sub_tokens.index('[CLS]')])
    print(DGembeddings[sub_tokens.index('[SEP]')])
    print(DGembeddings[sub_tokens.index('[MASK]')])
    print(DGembeddings[sub_tokens.index('happy')])

    DGembeddings_tensor = torch.tensor(DGembeddings, dtype=torch.float)
    torch.save(DGembeddings_tensor, r"./DG/DGembeddings.pth")



