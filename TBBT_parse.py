import csv, json

TRAIN_SPLIT_PERCENT = 0.8


# reading the CSV file
file= open('1_10_seasons_tbbt.csv', mode ='r', encoding='utf8')
csvFile = csv.reader(file)

data = []
for line in csvFile:
    data.append(line)
file.close()

# Load in vocab set
file = open('vocab_chars.json', 'r')
char_set = set(json.loads(file.read()))

#constuct text for episodes
episodes_text = []
current_episode = data[1][0]
text = ''
for i in range(1, len(data)):

    #check if episode is over
    if data[i][0] != current_episode:
        current_episode = data[i][0]

        #remove special charecters
        new=text
        for j in range(len(text)):
            if text[j].lower() not in char_set:
                new = new.replace(text[j],'')
                
        episodes_text.append('\n<episode>\n' + new + '\n</episode>\n')
        text = ''

    #build episode text data
    if data[i][2] != 'Teleplay' and data[i][2] != 'Story':
        text += data[i][2].lower() + ': ' + data[i][1].lower() +'\n\n'

#output data into txt file
file = open('tbbt_train.txt', 'w', encoding='utf8')
for episode in episodes_text[:int(len(episodes_text)*TRAIN_SPLIT_PERCENT)]:
    file.write(episode + '\n')
file.close()

file = open('tbbt_test.txt', 'w', encoding='utf8')
for episode in episodes_text[int(len(episodes_text)*TRAIN_SPLIT_PERCENT):]:
    file.write(episode + '\n')
file.close()

#print final info
print(f'{len(episodes_text[:int(len(episodes_text)*TRAIN_SPLIT_PERCENT)])} Episodes saved to training set, {len(episodes_text[int(len(episodes_text)*TRAIN_SPLIT_PERCENT):])} saved to test set')