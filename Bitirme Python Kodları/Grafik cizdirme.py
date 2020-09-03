import matplotlib.pyplot as plt

import os
prediction = []
label = []
f = open("Aynı sayıda paket.txt", "r")
counter =0
index =0
c_array = []
for x in f:
  counter = counter + 1
  if(counter % 20 == 0):
    index = index+1
    prediction.append(int(x))
    c_array.append(index*1.001)

print(len(prediction))
counter = 0;
label_counter = []
f1 = open("explosion026-t.txt", "r")
for y in f1:
    temp = y.split(' ')
    #print(temp[2])
    label.append(int(temp[2]))
    counter = counter+1
    label_counter.append(counter)

print(len(label_counter))
index = 0
equal_counter = 0
not_equal_counter = 0
for x in prediction:
    if(x == label[index]):
        equal_counter = equal_counter + 1
    else:
        not_equal_counter = not_equal_counter + 1
    index = index + 1
print("equal counter: " + str(equal_counter))
print("not equal counter: "+ str(not_equal_counter))

print("yuzde: " + str((100 * equal_counter) / (equal_counter+not_equal_counter)))
plt.plot(c_array, prediction)
plt.plot(label_counter, label)

plt.title('Videodaki anomali durumun tespiti')
plt.xlabel('Time \nMavi = Tahmin edilen, Turuncu = Gercek deger')
plt.ylabel('Class')

plt.show()
plt.savefig('my_figure.png')