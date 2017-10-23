import numpy as np
import sys
import gym

np.random.seed(0)


input_layer_size = 3
hidden_layer_size = 5
output_layer_size = 3


def pack(W1, b1, W2, b2):
    return np.concatenate([W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()])


def unpack(model):
    shapes = [
        (input_layer_size, hidden_layer_size),
        (1, hidden_layer_size),
        (hidden_layer_size, output_layer_size),
        (1, output_layer_size),
    ]
    result = []
    start = 0
    for i, offset in enumerate(np.prod(shape) for shape in shapes):
        result.append(model[start:start+offset].reshape(shapes[i]))
        start += offset
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def forward(model, x):
    W1, b1, W2, b2 = unpack(model)
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    return softmax(z2)


def finite_difference(f, model):
    numgrad = np.zeros(model.shape)
    perturb = np.zeros(model.shape)
    e = 1e-4
    for i in range(perturb.size):
        perturb.flat[i] = e
        loss1 = f(model - perturb)
        loss2 = f(model + perturb)
        numgrad.flat[i] = (loss2 - loss1) / (2 * e)
        perturb.flat[i] = 0
    return numgrad


def choose_action(action_distribution):
    r = np.random.random()
    total = 0
    for i, p in enumerate(action_distribution):
        total += p
        if r <= total:
            return i
class agent:
    def __init__(self):
        self.start_state=[0]*3
        self.curr_state=None
    def start_exp(self):
        self.curr_state=[0]*3 #self.start_state
        return 
    def nstate(self):
        return self.trans_state(self.curr_state)
    def curr_state(self):
        return trans_state(self.curr_state)
    def trans_state(self,state):
        state1= [ (i-5)/5 for i in state]
        return state1
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


def main():
    env=agent()
    c=0
    W1 = np.random.randn(input_layer_size, hidden_layer_size) / np.sqrt(input_layer_size)
    b1 = np.zeros((1, hidden_layer_size))
    W2 = np.random.randn(hidden_layer_size, output_layer_size) / np.sqrt(hidden_layer_size)
    b2 = np.zeros((1, output_layer_size))
    model = pack(W1, b1, W2, b2)

    discount_factor = 0.98
    gradient_step_size = 0.001
    rewards = []

    for episode in range(3000):
        env.start_exp()
        observation = env.nstate()

        observed_states = []
        executed_actions = []
        observed_rewards = []

        step = 0
        while True:
            step += 1

            observed_states.append(observation)
            action_distribution = forward(model, np.array(observation))[0]

            action = choose_action(action_distribution)

            observation, reward = env.do_exp(action)
            executed_actions.append(action)
            observed_rewards.append(reward)

            if env.is_end() or step > 3000:
                #print 'finished episode', episode, 'steps', step
                break
        steps = step

        discounted_rewards = observed_rewards * np.power(1, np.arange(steps))
        c=(c+1)%100
        if c==0:
            print(discounted_rewards.sum())
        rewards.append(discounted_rewards.sum())
        rewards = rewards[-100:]
        baseline = sum(rewards) / len(rewards)

        def log_policy(model):
            action_distribution = forward(model, np.array(observed_states))
            executed_action_probability = action_distribution[np.arange(len(executed_actions)), executed_actions]
            return np.sum(np.log(executed_action_probability))

        gradient_estimate = finite_difference(log_policy, model) * (discounted_rewards.sum() - baseline)
        # print 'gradient_estimate_magnitude', np.sqrt((gradient_estimate ** 2).sum())
        model += gradient_step_size * gradient_estimate
main()
