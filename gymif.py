import gym
import sys
from VRRL import VRRL,montocarloevaluation,policy_neural_net,sgd,bs1
import copy
from log import log

global name
name=None

class agent:
    def __init__(self):
        self.start_state=None
        self.curr_state=None
        self.env = gym.make(name)
        #self.env.monitor.start(outdir)
    def start_exp(self):
        observation=self.env.reset()
        self.curr_state=observation
        self.step=0
        self.flag=False
        return self.currstat()
    def currstat(self):
        return copy.deepcopy(self.curr_state)
    def do_exp(self,action):
        observation, reward, done, info = self.env.step(action)
        self.curr_state=observation
        self.step+=1
        if  self.flag or done or self.step > self.env.spec.timestep_limit:
            self.flag=True
        else:
            self.flag=False
        return self.currstat(),float(reward)
    def is_end(self):
        return self.flag

exp=['CartPole-v0','Acrobot-v1','MountainCar-v0','Pendulum-v0']
for k in exp:
    name=k
    for i in [True]:
        for j in [True,False]:
            envt=gym.make(name)
            input_layer_size = envt.observation_space.shape[0]
            hidden_layer_size = 16
            output_layer_size = envt.action_space.n
            tmlmt=envt.spec.timestep_limit
            del(envt)

            batch_size=64
            bs1=batch_size
            networkframework=[input_layer_size,8,8,output_layer_size]
            logp=log(name)

            ev=policy_neural_net(networkframework,batch_size)
            mc=montocarloevaluation(logp,batch_size=batch_size,baseline=i,vareduce=j)
            opt=sgd(0.2)
            vrrl=VRRL(ev,mc,bs1,agent,opt)
            vrrl.train(2000)

