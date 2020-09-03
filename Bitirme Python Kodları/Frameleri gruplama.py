import os

img_path = 'C:/Users/onure/PycharmProjects/ImageClassification/frames/'
f1 = open('explosion026.txt','r')

train_list = f1.readlines()

f3 = open('explosion026-t.txt', 'w')

clip_length = 20

counter_explosion = 0
counter_fighting = 0
counter_burglary = 0
counter_vandalism = 0
counter_arrest = 0

frame_counter = 0
frame_coefficient = 0
i = 0

for line in train_list:
    if(str(line)!='s'):
        name = line.split('F')[0]
        image_path = img_path+name
        label = line.split('\t')[-1]
        #images = os.listdir(image_path)
        temp1 = str(name)
        temp2 = str(train_list[i+1].split('F')[0])

        if (int(label) == 1):
            counter_explosion = counter_explosion + 1

        elif (int(label) == 2):
            counter_fighting = counter_fighting + 1

        elif (int(label) == 3):
            counter_burglary = counter_burglary + 1
            
        elif (int(label) == 4):
            counter_vandalism = counter_vandalism + 1    
        
        elif (int(label) == 5):
            counter_arrest = counter_arrest + 1
        
        if (temp1 != temp2 ):
            frame_counter = 0
            counter_explosion = 0
            counter_fighting = 0
            counter_burglary = 0
            counter_arrest = 0
            counter_vandalism = 0
            frame_coefficient = 0

        if(frame_counter == 20):
            if(counter_explosion >= 15):
                labell = 1
            elif (counter_fighting >= 15):
                labell = 2
            elif (counter_burglary >= 15):
                labell = 3
            elif (counter_vandalism >= 15):
                labell = 4
            elif (counter_arrest >= 15):
                labell = 5
            else:
                labell = 0
            f3.write(name + ' ' + str(frame_coefficient * clip_length + 1) + ' ' + str(labell) + '\n')
            print(name + ' ' + str(frame_coefficient * clip_length + 1) + ' ' + str(labell) + '\n')
            frame_coefficient = frame_coefficient + 1
            frame_counter = 0
            counter_explosion = 0
            counter_fighting = 0
            counter_burglary = 0
            counter_vandalism = 0
            counter_arrest = 0
    else:
        print('sona gelindi')
        break;
    frame_counter = frame_counter + 1
    i = i + 1

f1.close()
f3.close()
print("done")


