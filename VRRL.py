#encoding : utf-8

import copy
import numpy as np
import tensorflow as tf
import listlib as ll
import math
import time
from log import log

np.random.seed(0)#int(time.time()))

lrate=0.001
discount_factor=0.98
reg_opt=False
bs1=0
display_rate=20
cnt=0

def assert_positive_int(x):
    assert (isinstance(x,int))
    assert (x>0)
class neural_net_base:
    def __init__(self,network_layer,batch_size,stmfunc='sigmoid',outputfunc='softmax',initial_val='gauss'):
        """
        network_layer is a network sturucture list in which each dimension describe the number of layers.
        the first element denotes the input layer number, the last element denotes the output layer number
        initial_val option: 
            zero: zero
            gauss: Gaussian Distribution with \sigma=1\sqrt(input_number)
        """
        assert (isinstance(network_layer,list))
        assert (len(network_layer)>=2)
        assert_positive_int(batch_size)
        assert (initial_val in ['gauss','zero'])
        assert (stmfunc in ['relu','sigmoid','tanh'])
        self.batch_size=batch_size
        if stmfunc=='relu':
            self.stmfunc=tf.nn.relu
        elif stmfunc=='sigmoid':
            self.stmfunc=tf.sigmoid
        elif stmfunc=='tanh':
            self.stmfunc=tf.tanh
        self.initial_val=initial_val
        for i in network_layer:
            assert_positive_int(i)
        self.network_layer=network_layer
        self.input_num=network_layer[0]
        self.output_num=network_layer[-1]
        self.weight=[]
        self.bias=[]
        self.input=tf.placeholder(tf.float64,shape=(batch_size,self.input_num))
        self.llayer=len(network_layer)
        self.lweight=self.llayer-1
        self.layer=[]
        self.netupdate=[]
        self.netupdate_op=[]        
        for i in xrange(self.lweight):
            w=tf.Variable(self.new_net_variable([network_layer[i], network_layer[i+1]]),dtype=tf.float64)
            b=tf.Variable(tf.zeros([1,network_layer[i+1]],dtype=tf.float64),dtype=tf.float64)
            wup=tf.placeholder(tf.float64,shape=(network_layer[i], network_layer[i+1]))
            bup=tf.placeholder(tf.float64,shape=(1,network_layer[i+1]))
            self.weight.append(w)
            self.bias.append(b)
            self.netupdate.append([wup,bup])
            self.netupdate_op.append([w.assign_add(wup),b.assign_add(bup)])
            ''' May contains problem for dimension here check carefully'''
            if i==0:
                self.layer.append(self.stmfunc(tf.add(tf.matmul(self.input,self.weight[i]),self.bias[i])))
            else:
                self.layer.append(self.stmfunc(tf.add(tf.matmul(self.layer[i-1],self.weight[i]),self.bias[i])))
        self.construct_struct()
        self.sess=tf.Session()
        try:
            init = tf.global_variables_initializer()
        except:
            init= tf.initialize_all_variables()
        self.sess.run(init)

    def modify_struct(self):
        return None
    def update_struct(self,input):
        return None
    def evaluate(self,input):
        return None
    def new_net_variable(self,dim):
        assert (type(dim) is list)
        assert (len(dim) >= 1)
        if self.initial_val=='gauss':
            return np.random.randn(*dim)/ math.sqrt(float(dim[0]))
        elif self.initial_val=='zero':
            return tf.zeros(dim)
    def construct_struct(self):
        return None

class policy_neural_net(neural_net_base):
    def construct_struct(self):
        assert(len(self.layer)==self.lweight)
        self.prob=tf.nn.softmax(self.layer[self.lweight-1])
        self.logprob=tf.log(self.prob)
        self.gradients=[[None for j in xrange(self.output_num)] for i in xrange(self.batch_size)]
        for i in xrange(self.batch_size):
            for j in xrange(self.output_num):
                self.gradients[i][j]=[tf.gradients(self.logprob[i][j],self.weight),tf.gradients(self.logprob[i][j],self.bias)]
                #print(self.gradients[i][j])
        #stop()
        return
    def evaluate(self,input):
        ### need to refer to tf doc
        #print (input)
        #input=np.asarray(input)
        rst=self.sess.run((self.prob,self.gradients),feed_dict={self.input:input})
        prob=rst[0]
        grad=rst[1]
        #print(grad)
        return prob,grad
    def update(self,grads):
        #print(self.sess.run(self.weight[0]))
        for i in xrange(self.lweight):
            self.sess.run(self.netupdate_op[i][0],feed_dict={self.netupdate[i][0]:grads[0][i]})
            self.sess.run(self.netupdate_op[i][1],feed_dict={self.netupdate[i][1]:grads[1][i]})
#        self.opt.apply_gradients(upgrad)
        #self.sess.run(task)
        #print(self.sess.run(self.bias[i]))
        #print(self.sess.run(task))
            
class montocarloevaluation:
    def __init__(self,log,batch_size=1,lamb=2,baseline=False,vareduce=False):
        self.batch_size=batch_size
        self.sample_number=0
        self.lamb=lamb
        self.baseline=baseline
        self.vareduce=vareduce
        self.log=log
        self.totstep=0
        self.rewards=[]
        self.rq2=[]
        self.log.write("-----------------\nTime : %s; Base Line Option: %s; Variance Reduce Option: %s\n"
" Lambda: %f; Batch_size: %d; Discount_factor :%f; Learning_Rate: %f; Gradient Reg: %s\n-----------------\n"%(str(time.asctime(time.localtime(time.time()) )),str(self.baseline),str(self.vareduce),self.lamb,bs1,discount_factor,lrate,str(reg_opt)))
    def start_evaluate(self):
        self.sample_number=0
        self.edeltaq=None
        self.edeltaq2=None
        self.eq=0.0
        self.eq2=0.0
        self.bsl=None
        self.bs2l=None
        self.steps=0
    def evaluate(self,trajectories,gradients):
        """
        trajectories is list of trajectory
        trajectory is a list of tuple ( state , action , q)
        the last tuple should set action,q as None
        """
        #### pay attention ,replace all the following operations to tensorflow operation
        evalq2=[]
        #print(len(trajectories))
        ltrs=len(trajectories)
        self.rewards=self.rewards[-self.batch_size*4:]
        self.rq2=self.rq2[-self.batch_size*4:]
        for trs in xrange(ltrs):
            trajectory=trajectories[trs]
            #print(trajectory,"\n")
            assert (isinstance(trajectory,list))
            #print(len(trajectory))
            sumq=0
            df=1.0
            sumdelta=None
            ltr=len(trajectory)
            for tr in xrange(ltr):
                self.steps+=1
                rwd=trajectory[tr]
                assert (type(rwd) is float)
                sumq+=rwd*df
                sumdelta=ll.list2dsuma_f(gradients[trs][tr],sumdelta)
                df*=discount_factor
            self.sample_number+=1
            if self.baseline:
                if self.bsl==None:
                    if self.rewards==[]:
                        self.bsl=0.0
                    else:
                        self.bsl=(sum(self.rewards)/len(self.rewards))
                if self.bs2l==None:
                    if self.rq2==[]:
                        self.bs2l=0.0
                    else:
                        self.bs2l=(sum(self.rq2)/len(self.rq2))                                 
                self.rewards.append(sumq)
                sumq1=sumq-self.bsl
                if self.vareduce:
                    self.rq2.append(sumq*sumq)
                    sumq2=sumq*sumq-self.bs2l
            else:
                sumq1=sumq
                sumq2=sumq*sumq
            if self.vareduce:
                self.edeltaq2=ll.list2dsuma_f(ll.list2dmul_f(sumdelta,sumq2),self.edeltaq2)
            self.edeltaq=ll.list2dsuma_f(ll.list2dmul_f(sumdelta,sumq1),self.edeltaq)
            self.eq2+=sumq*sumq
            self.eq+=sumq
    def grad(self):
        global cnt
        if self.vareduce:
            alpha=2*math.sqrt(self.eq2/self.sample_number-(self.eq/self.sample_number)**2)
            if alpha>0:
                gamma=-self.lamb/(self.sample_number*alpha)
                beta=(self.eq*2*self.lamb)/((self.sample_number**2)*alpha)
                #print(gamma,beta)
        self.totstep+=1
        _flag=False
        if cnt>display_rate:  
            cnt=0
            _flag=True
        c=self.steps*1.0/self.sample_number
        if self.vareduce:
            a=self.eq/self.sample_number
            b=(self.eq/self.sample_number-self.lamb*math.sqrt(self.eq2/self.sample_number-(self.eq/self.sample_number)**2))
            _str="BatchNo: %d Reward: %f Goal: %f Step: %f \n" % (self.totstep,a,b,c)
            self.log.write(_str)
            if _flag:
                print(_str)
        else:
            a=self.eq/self.sample_number
            b=(self.eq/self.sample_number-self.lamb*math.sqrt(self.eq2/self.sample_number-(self.eq/self.sample_number)**2))
            _str="BatchNo: %d Reward: %f Variance: %f Step: %f\n" % (self.totstep,a,b,c)
            self.log.write(_str)
            if _flag:
                print(_str)
        if self.vareduce and alpha>0:
            return ll.list2dsuma_f( ll.list2dsuma_f(ll.list2ddiv_f(self.edeltaq,self.sample_number),ll.list2dmul_f(self.edeltaq2,gamma)),ll.list2dmul_f(self.edeltaq,beta))
        else:
            return ll.list2ddiv_f(self.edeltaq,self.sample_number)

class sgd:
    def __init__(self,learning_rate=0.001,grad_norm='normalize_one'):
        global lrate
        lrate=learning_rate
        self.cnt=0
        self.learning_rate=learning_rate
        assert (grad_norm in ['normalize_one','normalize_max',None])
        if grad_norm=='normalize_one':
            self.grad_norm=self.normalize_one
        elif grad_norm=='normalize_max':
            self.grad_norm=self.normalize_max
        elif grad_norm=='None':
            self.grad_norm=self.nothing
    def nothing(self,grads):
        return grads
    def normalize_max(self,grads):
        tot=None
        for i in grads:
            for j in i:
                if tot==None:
                    tot=np.max(j)
                else:
                    tot=max(tot,np.max(j))
        if tot<1e-3:
            return grads
        else:
            return [[j / tot for j in i] for i in grads]
    def normalize_one(self,grads):
        tot=0
        for i in grads:
            for j in i:
                tot+=np.sum(np.square(j))
        tot=math.sqrt(tot)
        if tot<1e-3:
            return grads
        else:
            return [[j / tot for j in i] for i in grads]
    def opt(self,grads):
        self.cnt+=1
        if self.cnt>=100 and self.learning_rate>0.01:
            self.learning_rate*=0.7
            self.cnt=0
        return ll.list2dmul_f(self.grad_norm(grads),self.learning_rate)
    

class VRRL: #VarianceReducedReinforcementLearning
    def __init__(self,policy_net,evaluation_net=None,batch_size=1,agent_object=None,optimizer=sgd()):
        """
        Policy Net gives a policy estimator
        Evaluation net gives a Q and Q^2 function estimator, when it's None , Monto Carlo is used to sample this value
        void start_exp() marks the agents start to do a new experiment from start_point
        do_exp(action) returns {new state,reward}, it let the agent to do specific action at the experiment, new state equals None denoting end states
        is_end() return whether exp is end
        """
        #self.tracker = SummaryTracker()  
        self.trajectories=[]
        self.policy_net=policy_net
        self.evaluation_net=evaluation_net
        #self.start_exp=agent_func["start_exp"]
        self.agent_object=agent_object
        self.net_batch_size=policy_net.batch_size 
        self.action_size=policy_net.output_num
        self.numberofbatch=((batch_size-1)//self.net_batch_size)+1
        self.optimizer=optimizer
    def stochasticaction(self,action_prob):
        x=np.random.random()
        _sum=0.0
        for i in xrange(self.action_size):
            _sum+=action_prob[i]
            if _sum>=x:
                return i
        return self.action_size-1
    def train(self,step):
        global cnt
        agents=[None for i in xrange(self.net_batch_size)]
        prev_states=[None for i in xrange(self.net_batch_size)]
        for i in xrange(self.net_batch_size):
            agents[i]=self.agent_object()
        for sid in xrange(step):
            #if sid%100==0:
            #    self.tracker.print_diff()
            self.evaluation_net.start_evaluate()
            for bid in xrange(self.numberofbatch):
                trajectories=[[] for i in xrange(self.net_batch_size)] #check this code
                gradients=[[] for i in xrange(self.net_batch_size)]
                end_traj=False
                for sbid in xrange(self.net_batch_size):
                    prev_states[sbid]=agents[sbid].start_exp()
                stepsize=0
                cnt+=self.net_batch_size
                while not end_traj:
                    stepsize+=1
                    action_vec,action_grad=self.policy_net.evaluate(prev_states)
                    end_traj=True
                    for sbid in xrange(self.net_batch_size):
                        if not agents[sbid].is_end():
                            action_id=self.stochasticaction(action_vec[sbid])
                            ac_gr=copy.deepcopy(action_grad[sbid][action_id])
                            gradients[sbid].append(ac_gr)#action_grad[sbid][action_id])
                            n_state,reward = agents[sbid].do_exp(action_id)
                            trajectories[sbid].append(reward)
                            prev_states[sbid]=n_state#copy.deepcopy(n_state)
                            end_traj=False
                #print("stepsize:%d"%(stepsize))
                #if sid==9:
                #print trajectories
                self.evaluation_net.evaluate(trajectories,gradients)                
            #if cnt>display_rate:
                #print (self.policy_net.sess.run(self.policy_net.weight))
            grads=self.evaluation_net.grad()
            self.policy_net.update(self.optimizer.opt(grads))

class agent:
    def __init__(self):
        self.start_state=[0]*3
        self.curr_state=None
    def start_exp(self):
        self.curr_state=[0]*3 #self.start_state
        return self.currstat()
    def trans_state(self,state):
        state1= [ (i-5)/5.0 for i in state]
        return state1
    def currstat(self):
        return self.trans_state(self.curr_state)
    def do_exp(self,action):
        self.curr_state[action]+=1
        rwd=0
        for i in self.curr_state:
            rwd+=i*i
        return self.trans_state(self.curr_state),float(rwd)
    def is_end(self):
        flag=False
        for i in self.curr_state:
            if i>=10:
                flag=True
        return flag

if __name__=="__main__":
    global ev
    print ("into main")
    batch_size=64
    nob=1
    logp=log("3Number")
    networkframework=[3,8,8,3]
    bs1=batch_size*nob

    for i in [True]:
        for j in [True,False]:
            ev=policy_neural_net(networkframework,batch_size)
            opt=sgd(0.2)
            mc=montocarloevaluation(logp,batch_size=batch_size,baseline=i,vareduce=j)
            vrrl=VRRL(ev,mc,bs1,agent,opt)
            vrrl.train(2000)
