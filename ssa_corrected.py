import random
import math
import numpy as np
dimension=random.randint(5,10)*2

def object_func(arr):
  sign=1
  length=int(len(arr)/2)
  ans=0
  for i in range(length):
    ans+=sign*math.fabs(arr[2*i]-arr[2*i+1])
    sign*=-1
  return math.fabs(ans)

def ssa(obj,l,u,dim,n,max_it):
  salp=np.zeros((n,dim))# n is the number of salps
  # print(salp.shape)
  for i in range(dim):
    #salp[i]=np.random.uniform(l[i],u[i],dim)#randomly initializing the salps
    salp[:, i] = np.random.uniform(0, 1, n) * (u[i] - l[i]) + l[i]
  #calculating the fitness of each salp
  primary_fitness=np.zeros(n)
  for i in range(n):
    primary_fitness[i]=obj(salp[i,:])
  temp_key=np.argsort(primary_fitness)
  salp=salp[temp_key,:]#sorting the salps to put the leader in the front
  #primary_fitness.sort()#sorting the primary_fitness
  food_pos=salp[0,:]
  food_fitness=np.min(primary_fitness)
  it=1
  d1=2*math.exp(-((4*it/max_it)**2))
  while it<=max_it:
    d2=random.randint(0,100)/100
    d3=random.randint(0,200)/100-1
    #updation of the salps
    for i in range(n):
      if i==0:
        for j in range(dim):
          if d3>=0:
            salp[i][j]=food_pos[j]+d1*((u[j]-l[j])*d2+l[j])#bounds are dim-wise
          else:
            salp[i][j]=food_pos[j]-d1*((u[j]-l[j])*d2+l[j])
      else:
        salp[i,:]=(salp[i,:]+salp[i-1,:])/2
    for i in range(n):
      #clipping
      for j in range(dim):
        salp[i,j]=np.clip(salp[i,j],l[j],u[j])
      
      temp_salp_fitness=obj(salp[i,:])
      # print(temp_salp_fitness)
      if temp_salp_fitness<food_fitness:
        food_fitness=temp_salp_fitness
        food_pos=salp[i]
    print(f"for {it}: fitness was {food_fitness}")
    it+=1
  return food_fitness

upper_bound=np.zeros(dimension)
lower_bound=np.zeros(dimension)

for i in range(dimension):
  lower_bound[i]=random.randint(20,30)
  upper_bound[i]=lower_bound[i]+random.randint(20,30)
#salp swarm
ssa(object_func,lower_bound,upper_bound,dimension,30,30)
