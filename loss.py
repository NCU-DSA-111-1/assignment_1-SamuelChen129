#use anaconda powershell for execution
import matplotlib.pyplot as plt
file=open("loss.txt")
data=file.read()
n=[]
loss=[]
read_n=0
read_loss=0
read=""
plt.figure()
for word in data:
    read+=word
    if read_n==0 and read=="n=":
        read=""
        read_n=1
    if read_loss==0 and read=="loss=":
        read=""
        read_loss=1
    if word==" " and read_n==1:
        n.append(int(read))
        read_n=0
        read=""
    if word=="\n" and read_loss==1:
        loss.append(float(read)) 
        read_loss=0
        read=""
plt.plot(n, loss)
plt.xlabel("training times")
plt.ylabel("loss")       
plt.show()    
file.close()
