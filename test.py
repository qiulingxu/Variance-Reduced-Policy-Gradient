import os
import random
a=[0,0,0]
ans=0
dp=0.98
dc=1.0
for j in xrange(3):
    for i in xrange(9):
        a[j]+=1
        for k in xrange(3):
            ans+=a[k]*a[k]*dc
        dc*=dp
print(ans)

ans=0.0
for tp in xrange(10000):
    dp=0.98
    dc=1.0
    a=[0,0,0]
    f=True
    for j in xrange(30):
            a[random.randint(0,2)]+=1
            for k in xrange(3):
                if a[k]>9:
                    f=False
                    break
            if not f:
                break
            for k in xrange(3):
                ans+=a[k]*a[k]*dc
            dc*=dp
print(ans/10000)
