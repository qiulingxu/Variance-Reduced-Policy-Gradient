import copy
def listop2(*arg,**kwargs):
    argn=len(arg)
    listlen=0
    ll=[]
    op=kwargs["op"]
    t=-1
    for i in xrange(argn):
        assert(isinstance(arg[i],list) or isinstance(arg[i],float) or isinstance(arg[i],int))
        if isinstance(arg[i],list):
            xl=len(arg[i])
            if (t!=-1 and t!=xl):
                assert(True)
            t=xl
        else:
            xl=-1
        ll.append(xl)
        listlen=max(listlen,xl)
    rst=[None for i in xrange(listlen)]
    for i in xrange(argn):
        tn=None
        if ll[i]==-1:
            l=listlen
            x=arg[i]
        else:
            l=ll[i]
        for j in xrange(l):
            if ll[i]!=-1:
               x=arg[i][j]
            if rst[j] is None:
                rst[j]=copy.deepcopy(x)
            else:
                rst[j]=op(rst[j],x)
    return rst

def listop1(lst,op):
    assert(isinstance(lst,list))
    rst=[]
    for i in lst:
        rst.append(op(i))
    return rst

def sum1(a,b):
    return a+b

def mul(a,b):
    return a*b

def div(a,b):
    return a/b


def listsum(*arg):
    return listop2(*arg,op=sum1)

def listmul(*arg):
    return listop2(*arg,op=mul)

def listdiv(*arg):
    return listop2(*arg,op=div)

def list2dsum(*arg):
    return listop2(*arg,op=listsum)

def list2dmul_f(a,b):
    return [[j * b for j in i ] for i in a]

def list2ddiv_f(a,b):
    return [[j / b for j in i ] for i in a]

def list2dsum_f(a,b):
    return [[j + b for j in i ] for i in a]

def list2dsuma_f(a,b):
    if b is None:
        return a
    l1=len(a)
    l2=len(b)
    if (l1!=l2):
        print(l1,l2)
        assert(False)
    rst=[[] for i in xrange(l1)]
    for i in xrange(l1):
        ls1=len(a[i])
        ls2=len(b[i])
        assert(ls1==ls2)
        for j in xrange(ls1):
            rst[i].append(a[i][j]+b[i][j])
    return rst

def list2dmul(*arg):
    return listop2(*arg,op=listmul)

def list2ddiv(*arg):
    return listop2(*arg,op=listdiv)

if __name__=="__main__":
    print ("into test of listlib")
    a=[[2.0,3.0,2.0,2.0], [4.0, 5.0], [3.0]]
    b=[[3.0, 2.0], [6.0, 7.0]]
    c=[[3.5, 2.5], [6.5, 7.5]]
    print("a:"+str(a))
    print("b:"+str(b))
    """for i in xrange(100):
        c=listsum(a,b)
    print ("listsum(a,b):"+str(listsum(a,b)))
    print ("list2dsum(a,b):"+str(list2dsum(a,b)))
    print ("list2dsum(0,b):"+str(list2dsum(0,b)))
    print ("list2dmul(a,b):"+str(list2dmul(a,b)))
    print ("list2ddiv(a,b):"+str(list2ddiv(a,b)))
    print ("list2dsum(a,2.0):"+str(list2dsum(a,2.0)))
    print ("list2dmul(a,2.0):"+str(list2dmul(a,2.0)))
    print ("list2ddiv(a,2.0):"+str(list2ddiv(a,2.0)))"""

    print ("list2dsuma_f(c,b):"+str(list2dsuma_f(c,b)))
