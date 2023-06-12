import numpy as np

files = np.load('./DHN/mirflickr0.8026341532286411-trn_binary.npy')


file = files.tolist()

for i in range(len(file)):
    for j in range(len(file[0])):
        if file[i][j] == 1:
            file[i][j] = 1
        else:
            file[i][j] = 0

def tostr(Nums):
    strNums = [str(x) for x in Nums]
    return "".join(strNums)

with open("pool.txt", "a") as f1:
    for i in range(len(file)):
        f1.writelines(tostr(file[i]) + '\n')



# np.savetxt('hashcode.txt',file)

