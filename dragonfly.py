from Modal.function.function import *

import math
import numpy as np
import random

from Modal.function.function.get_Function import get_function_details
class dragonfly:
  def Levy(self,dim):
    beta=1.5
    sigma=((math.gamma(1+beta)*math.sin(math.pi*beta/2))/(math.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
    r1=random.random()
    r2=random.random()
    return 0.01*((r1*sigma)/(r2**(1/beta)))

  def dist_of_all_dim(self,dim,arr1,arr2):
    ans=np.zeros((dim))
    for i in range(0,dim):
      ans[i]=math.fabs(arr1[i]-arr2[i])
    return ans
  
  def is_neighbour(self,dim,arr1,arr2,r):
    same=0
    #checks over all the dimensions and sees if it is in range and also sees if all
    #the locations are exactly same. If so, then returns false otherwise true.
    for i in range(0,dim):
      diff_val=math.fabs(arr1[i]-arr2[i])
      # print(f"type of diff_val: {type(diff_val)}")
      # print(f"type of r: {type(r)}")
      if diff_val>r[i]:
        return False
      if diff_val==0:
        same+=1
    if(same==dim):
      return False
    else:
      return True

  def initialize(self,n,dim,ub,lb):
    ans=np.zeros((n,dim),dtype="float")
    #randomly generating all the elements
    for i in range(0,n):
      for j in range(0,dim):
        #point to note: rand(a,b) means [a,b] both ranges inclusive
        ans[i][j]=(random.random())*(ub[j]-lb[j])+lb[j]
    return ans

  def find_neighbours(self,pop,agent,r):
    neighbours = []
    for i in range(np.shape(pop)[0]):
        if self.is_neighbour(np.shape(pop)[1],agent,pop[i],r):
            neighbours.append(i)
    neighbour_list = np.zeros(len(neighbours),dtype="int")
    for i in range(len(neighbours)):
        neighbour_list[i] = neighbours[i]
    return neighbour_list
    #fixed bugs but still not tested as a whole
  def altruism(self,obj,dragonflies,rang,food,f_factor,enemy,e_factor,lb,ub):

      agents = dragonflies.copy()
      rows,cols = np.shape(agents)
      
      #find altruist i.e., the worst dragonfly
      altruist = 0
      altruist_fit = obj(agents[altruist])
      for i in range(1,rows):
        if obj(agents[i]) > altruist_fit:
          altruist = i
          alturist_fit = obj(agents[i])

      #find all neighbours of altruist
      """neighbours = []
      for i in range(rows):
        if i == altruist:
          continue
        if self.is_neighbour(cols,agents[altruist],agents[i],rang):
          neighbours.append(i)"""
      neighbours = self.find_neighbours(agents,agents[altruist],rang)

      if len(neighbours) == 0:
        return dragonflies
      
      #find beneficiary
      beneficiary = np.random.randint(0,len(neighbours))

      fitness_alt = altruist_fit
      # fitness_ben = obj(beneficiary)
      fitness_ben=obj(agents[beneficiary])

      #find Hamilton equation's factor
      distance = np.zeros(cols)
      for i in range(cols):
        distance[i] = abs(agents[altruist][i]-agents[beneficiary][i])

      #closer the beneficiary, lesser the scaling required
      #second part will always be less than 1
      r = 1 - np.sum(distance)/np.sum(rang)

      #generate a random number for the food offset
      dist1 = self.dist_of_all_dim(cols,food,agents[altruist])
      dist2 = self.dist_of_all_dim(cols,food,agents[beneficiary])
      dist3 = self.dist_of_all_dim(cols,enemy,agents[altruist])
      dist4 = self.dist_of_all_dim(cols,enemy,agents[beneficiary])

      for i in range(cols):

        offset = np.random.uniform(low=0,high=min(abs(dist1[i]),abs(dist2[i]),abs(rang[i])))

        #need not change delta as it will anyways change on next iteration
        agents[altruist][i] += (f_factor*offset)
        agents[beneficiary][i] -= (f_factor*offset)

        offset = np.random.uniform(low=0,high=min(abs(dist3[i]),abs(dist4[i]),abs(rang[i])))

        #need not change delta as it will anyways change on next iteration
        agents[altruist][i] -= (e_factor*offset)
        agents[beneficiary][i] += (e_factor*offset)

      #perform clipping
      for i in range(cols):
        agents[altruist][i] = min(agents[altruist][i],ub[i])
        agents[altruist][i] = max(agents[altruist][i],lb[i])
        agents[beneficiary][i] = min(agents[beneficiary][i],ub[i])
        agents[beneficiary][i] = max(agents[beneficiary][i],lb[i])

      #check if altruism is successful
      # benefit = obj(beneficiary)
      benefit=obj(agents[beneficiary])
      # cost = obj(altruist)
      cost=obj(agents[altruist])
      if benefit-fitness_ben <= 0 and cost-fitness_alt >= 0:
        benefit = fitness_ben - benefit
        cost = cost - fitness_alt
      else:
        return dragonflies
      if r*benefit > cost:
        return agents
      else:
        return dragonflies

  def optimize(self,obj,dim,bound,n,max_it):
    #getting the dimensions
    # dim=np.size(bound)[2]
    ub=np.zeros((dim),dtype="float")
    lb=ub.copy()
    
    #setting the values of the range
    for i in range(0,dim):
      lb[i]=bound[i][0]
      ub[i]=bound[i][1]
    
    #setting the initial r. In paper code, this magic_r_val was 10
    magic_r_val=10
    r=(ub-lb)/magic_r_val
    delta_max=(ub-lb)/magic_r_val
    
    #should I set the best and worst fitness as the enemy and food respectively at the begining???

    #setting the enemy and food fitness
    food_fitness=np.inf
    enemy_fitness=-np.inf
    #setting the initial food and enemy position
    #maybe change it to better suit the needs
    food_pos=np.zeros((dim),dtype="float")
    enemy_pos=np.zeros((dim),dtype="float")
    
    #initializing the dragonflies X aka positions
    x=self.initialize(n,dim,ub,lb)
    #probably change this
    delta_x=self.initialize(n,dim,ub,lb)

    #main loop
    for iter in range(1,max_it+1):

      #a lot of magic numbers here
      r=(ub-lb)/4+((ub-lb)*(iter/max_it)*2)

      w=0.9-iter*((0.9-0.4)/max_it)
        
      my_c=0.1-iter*((0.1-0)/(max_it/2))
      if(my_c<0):
        my_c=0
      s=2*(random.random())*my_c
      a=2*(random.random())*my_c
      c=2*(random.random())*my_c
      f=2*(random.random());     
      e=my_c
      fitness_prev=np.zeros((n))
      #calculating the fitness values of the dragonflies
      for i in range(0,n):
        cur_fitness=obj(x[i,:])
        fitness_prev[i]=cur_fitness
        print(f"cur_fitness for {i} with {x[i,:]}: {cur_fitness}")
        #possible confusion about bound checking here
        #i deemed that not useful
        if cur_fitness>enemy_fitness:
          enemy_fitness=cur_fitness
          enemy_pos=x[i,:].copy()
        if cur_fitness<food_fitness:
          print(f"for updation: {cur_fitness}, {food_fitness}")
          food_fitness=cur_fitness
          food_pos=x[i,:].copy()
      print(f"in loop {iter}, fit: {food_fitness}, loc: {food_pos}")
      #updating the dragonflies one by one
      for i in range(0,n):
        # #calculating the neighbours
        # total_neighbour=0
        # neighbour_index=np.zeros((n),dtype="int")
        # for j in range(0,n):
        #   if self.is_neighbour(dim,x[i,:],x[j,:],r):
        #     neighbour_index[total_neighbour]=j
        #     total_neighbour+=1
        neighbour_index = self.find_neighbours(x,x[i],r)
        total_neighbour = len(neighbour_index)
        #calculating the s,c,a,f,e parameters
        #possible confusion:
        #the default case should be done when number of neighbours ==0 ( but it is given as 1 in the paper code)

        #s
        #huge problem in the S calculation paper says x[k]-x but code from other source says x-x[k]
        Separation=np.zeros((dim),dtype="float")
        for k in range(0,total_neighbour):
          # print(f"type of nei_index: {np.shape(neighbour_index)}")
          Separation+=x[neighbour_index[k],:]-x[i,:]
        #a
        Alignment=np.zeros((dim),dtype="float")
        if total_neighbour>0:
          for k in range(0,total_neighbour):
            Alignment+=delta_x[neighbour_index[k],:]
          Alignment/=total_neighbour
        else:
          Alignment=delta_x[i,:].copy()
        #c
        Cohesion=np.zeros((dim),dtype="float")
        if total_neighbour>0:
          for k in range(0,total_neighbour):
            Cohesion+=x[neighbour_index[k],:]
          Cohesion/=total_neighbour
        else:
          Cohesion=x[i,:].copy()
        Cohesion-=x[i,:]
        #f
        dist_to_food=self.dist_of_all_dim(dim,food_pos.copy(),x[i,:])
        Food_attr=np.zeros((dim),dtype="float")
        #checking if food is in reach
        if np.shape(np.where(dist_to_food<=r))[1]==dim: #careful with the expr
          Food_attr=food_pos.copy()-x[i,:]
        #e
        dist_to_enemy=self.dist_of_all_dim(dim,enemy_pos.copy(),x[i,:])
        Enemy_attr=np.zeros((dim),dtype="float")
        #checking if enemy is in reach
        if np.shape(np.where(dist_to_enemy<=r))[1]==dim: #careful with the expr
          Enemy_attr=enemy_pos.copy()+x[i,:]
        
        #clipping but it looks very fishy. because >ub becomes lb and <lb becomes ub
        for j in range(0,dim):
          if x[i,j]>ub[j]:
            x[i,j]=lb[j] #check
            delta_x[i,j]=random.random()
          if x[i,j]<lb[j]:
            x[i,j]=ub[j] #check
            delta_x[i,j]=random.random()
        
        #checking to see if any food_dimension in out of reach
        if np.shape(np.where(Food_attr>r))[1]>0: #careful with the expr
          if total_neighbour>0:
            for j in range(0,dim):
              delta_x[i,j]=w*delta_x[i,j]+(random.random())*Alignment[j]+(random.random())*Cohesion[j]+(random.random())*Separation[j]
              if delta_x[i,j]>delta_max[j]:
                delta_x[i,j]=delta_max[j]
              if delta_x[i,j]<-delta_max[j]:
                delta_x[i,j]=-delta_max[j]
              x[i,j]+=delta_x[i,j]
          else:
            x[i,:]+=self.Levy(dim)*x[i,:]#need to implement levy
            delta_x[i,:]=np.zeros((dim),dtype="float")
        else:
          for j in range(0,dim):
            delta_x[i,j]=(a*Alignment[j]+c*Cohesion[j]+s*Separation[j]+f*Food_attr[j]+e*Enemy_attr[j])+w*delta_x[i,j]
            if delta_x[i,j]>delta_max[j]:
              delta_x[i,j]=delta_max[j]
            if delta_x[i,j]<-delta_max[j]:
              delta_x[i,j]=-delta_max[j]
            x[i,j]+=delta_x[i,j]

        
        #temporary clipping
        for j in range(0,dim):
          if x[i,j]>ub[j]:
            x[i,j]=ub[j]
          if x[i,j]<lb[j]:
            x[i,j]=lb[j]
        # print(f"cur x val for {i} : {x[i,:]}")


        #there were these 3 lines of matlab code that I could not decipher:
        #didn't understand what they do

        # Flag4ub=X(:,i)>ub';
        # Flag4lb=X(:,i)<lb';
        # X(:,i)=(X(:,i).*(~(Flag4ub+Flag4lb)))+ub'.*Flag4ub+lb'.*Flag4lb;
      
      #if altruism worked
      temp_pop=self.altruism(obj,x,r,food_pos,f,enemy_pos,e,lb,ub)
      if(min([obj(i) for i in temp_pop])<min([obj(i) for i in x])):
        x=temp_pop.copy()

      #altruism call
      #opportunistic altruism 
      # temp_pop=self.altruism(obj,x,r,food_pos,f,enemy_pos,e,lb,ub)
      # fitness_new=np.zeros((n))
      # for t in range(0,n):
      #   fitness_new[t]=obj(temp_pop[t,:])
      #   # fitness_prev[t]=obj(x[t,:])
      # if(min(fitness_new)<min(fitness_prev)):
        # x=temp_pop
      #
      #x = self.altruism(obj,x,r,food_pos,f,enemy_pos,e,lb,ub)
    return food_pos,food_fitness #returning loc and fit      


import pandas as pd
points=[]
for i in range(5):
  max_iter=100
  pop_size=20
  name='F'+str(i+1)
  fobj, lb, ub, dim=get_function_details(name)
  drag=dragonfly()
  loc,fit=drag.optimize(fobj,dim,zip(lb,ub),pop_size,max_iter)
  points.append((name,loc,fit))
df=pd.DataFrame(points,columns=["name","x","y"])
df.to_csv("unimodal_csv.csv")
print(df)