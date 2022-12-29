#%%
x1 = 0 
v1 = 3 

x2 = 4
v2 = 2

def kangaroo(x1, v1, x2, v2):
    if(v1 > v2):
        if (((x2-x1)/(v1-v2)) %2) == 0:
            return "YES"
    return "NO"
        
    

        

kangaroo(x1, v1, x2, v2)