import os
import random
root_path = "/home/raoblack/Documents/create_dataset/mnt_small/ramdisk/max/90kDICT32px"


total_data_num = 0
#create imlist.txt
with open("imlist.txt", 'w') as out_f:
    for r, d, fs in os.walk(root_path):
        #print("r=", r)
        #print("d=", d)
        for f in fs:
            if ".jpg" in f:
                rel_path = "./"+os.path.relpath(os.path.join(r, f), root_path)
                #print("relative path of {} is {}".format(f, rel_path))
                out_f.write(rel_path+"\n")
                total_data_num += 1
print("total data number is {}".format(total_data_num))

#create annotation.txt

with open("imlist.txt") as in_f, open("annotation.txt", 'w') as out_f:
    for line in in_f:
        file_name = os.path.basename(line)
        file_n, _ = file_name.split('.')
        _, _, serial_num = file_n.split('_')
        # print(file_n)
        # print(serial_num)
        new_line = line[:-1] + " " + serial_num
        out_f.write(new_line + "\n")

train_ratio = 0.8
test_ratio = 0.1
val_ratio = 1.0 - train_ratio - test_ratio

train_num = round(total_data_num*train_ratio)
test_num = round(total_data_num*test_ratio)
val_num = total_data_num - train_num - test_num

#create annotation_train.txt
with open("annotation.txt") as in_f, open("annotation_train.txt", 'w') as out_f1, \
    open("annotation_test.txt", 'w') as out_f2, open("annotation_val.txt", 'w') as out_f3:
    shuffled_data = [ (random.random(), line) for line in in_f ]
    shuffled_data.sort()
    count = 0
    for _, line in shuffled_data:
        if 0 <= count and count < train_num:
            out_f1.write(line)
        if train_num <= count and count < (train_num + test_num):
            out_f2.write(line)
        if (train_num + test_num) <= count and count < total_data_num:
            out_f3.write(line)
        count += 1



#create annotation_test.txt



#create annotation_val.txt
