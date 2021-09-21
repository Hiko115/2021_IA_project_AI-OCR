import pandas as pd
import re


temp = ['WOISOnS', 's7efi （HKDN', '傑49.90', 'YAS', '18/07/30 21:36:10', 'S3235', '97%t', 'QQ1467']

save = pd.Series(temp)


for i in temp:
    if i is temp[0]:
        b = i       
    if "/" in i:
        t = i
    if "$" in i:
        m = i
    if "." in i:
        m = i
    if "HKD" in i:
        p =i
    
            
print(save)

