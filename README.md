# Variance-Reduced-Policy-Gradient
A research project for enhancing stability of Policy Gradient Algorithm

## Research Path

### STUDY WEEK Report #2

> 20161106 Jerry

#### Done List

-

- Propose VCQL and VCSARSA
- Go through 2 Papers

Unrelated Stuff:

1. Learn how to use Haskell : A functional programming language To Chapter 12
2. Revise for Physics Mid-Term
3. restore from sick

#### Paper Note

--------------------------------------------------------------------------------

#### Mean-Variance Optimization in Markov Decision Processes

> 2011 posted 2013 accepted , John N. Tsitsiklis , MIT , Computer Science Field

#### Content

1. Author define a computer science problem for computing a fully observable mdp whose $Variance < C_1$ and $Q(s,a)> C_2$.
2. Author prove that, By **Constructing Method**, the specific subproblem can be reducted to **Subset Sum Problem** , Which indicates the subproblem is **NP-Hard**.
3. Author gives a **useless** approximate algorithm which make reductions of this problem into Interger Linear Programming.

#### Thoughts

1. We can also show one specific subproblem is actually P-Hard, Thus **whether the total problem is NP-Hard is under doubt**.
2. This point hasn't be applied into Reinforcement Learning yet.
3. The technique of appoximation to converse the limit equations into a **polyhedron** and then apply **Interger Linear Programming**. However this method is totally useless for any problem. Not only because it requires every information like probability transition, but also the complexity is actually exponential for real problem due to its unrealistic constraints in this paper.

--------------------------------------------------------------------------------

#### Quantile Reinforcement Learning

> 2016 Hugo Gilber, Sorbonnes Universit´es; Paul Weng , CMU , Machine Learning Field

#### Content

1. The author propose a new criterior for MDP that based on **order** of result.
2. They design a new problem which maimize the worst case with hisgh probalbility getting an better endstate. 3\. They only consider the speical MDP, only with last-step reward.
3. They transverse the solution to a constucted MDP with Q-Learning given the worst case.
4. They design a new algorithm , simultaneously optimize the worst case and the best policy under that worst case based on paper [1].

#### Thoughts

1. Their proposed algorithm fully relies on the **last-step reward**, which actually makes no contributions to the real problem. Because for non-order reward this technique also works and is trivial.
2. The 2 Timescale technique is useful. Supposing we have problem $max(u(v(x))) \forall u,v$ we now only need to make sure $|log(\frac{u_t+1}{u_t})|<|log(\frac{v_t+1}{v_t})|$ and the convergence of $max(u(constant))$ and we can prove simultaneous optimizing these 2 function will eventually works.
3. The idea is interesting because we now only consider the order of the reward.
4. Robustness idea contains in their work.

[1]: V.S. Borkar. Stochastic approximation with time scales. Systems & Control Letters, 29(5):291–294, 1997

### Study Week Report 5

#### Paper Review

##### Regularized Policy Gradients: Direct Variance Reduction in Policy Gradient Estimation

- ACML 2015

##### Author

- Tingting Zhao & Jucheng Yang, Tianjin University of Science and Technology
- Gang Niu, The University of Tokyo
- Ning Xie,Tongji University

##### Content

- the author stress that in policy gradient , the **estimate of gradient** is of high variance. So they propose a direct method to decrease the vairance, that put the variance of gradient into evalution function as a punishment term. The common policy gradient loss function is $E(R_\tau)$ , and they change it into $E_\theta[R_\tau]-\lambda\cdot Var(\nabla_\theta E_\theta[R_\tau])$.
- Author design three experiments to testify the algorithm : A manmade function , Mountain Car , Stoke Based Rendering System

##### Thoughts

- its aim is to decrease the variance of the gradient estimate, and may have similar effect as directly decrease variance.
- the last experiment , **Stoke Based Rendering System**, is impressive

#### Two methods for policy gradient

##### Solving Linear equation

- $f(x)=f(x_s)+(x-x_s)\cdot\nabla{f(x_s)}+o(x-x_s)^2$ when $x->x_s$ we can take it as $f(x)=f(x_s)+(x-x_s)\cdot\nabla{f(x_s)}$. Thus to calculate $\nabla f(x)$ we just need to get $f(x_s+\delta_i)$ for n times. and we can get $n$ equations with the form $f(x_s+\delta_i)-f(x_s)=\nabla\cdot{\delta_i}$. calculate its inverse is suffice to get its gradient.

##### Sampling

- to calculate $\nabla(\int_\tau p_\theta(\tau)\cdot R_\tau)=\int_\tau p_\theta(\tau)\cdot \nabla log(p_\theta(\tau))\cdot R_\tau=E( log(p_\theta(\tau))\cdot R_\tau)$. And just to sample according this formula to get the gradient.

#### Proposed Method

##### Optimize with hard constraint

First We try to use the KKT to convert the variance limit but failed.

Formalization consider the start point $s_0$, $Max(E(R_{s_0}))$ s.t. $Var(R_{s_0})\le C$. By KKT of lagarian , we can have the equal form of set of equation

$\nabla[E(R_{s_0})-\lambda(Var(R_{s_0})-C)]=0 ,$ $\lambda\cdot [Var(R_{s_0})-C]=0 ,$ $Var(R_{s_0})-C\le 0$

and if we limit the answer to be interior of the possible solution we can relax the condition to $\nabla E(R_{s_0})=0,$ $Var(R_{s_0})-C< 0$ However it's not trivial to solve $\nabla E(R_{s_0})=0$.

##### Penalty Item soft constriant

Then we consider the soft bound

**Notice that different from ordinary Reinforcement Learning, we are now specifically optimizing $E(R_{s_0})$ and $Var(R_{s_0})$ $s_0$ is the start point, and the other states is of no concern to us.**

Now we try to optimize $max(E(R_{s_0})-\lambda\sqrt{E(R_{s_0}^2)-E(R_{s_0})^2})$

Given a $\epsilon$-greedy $\theta$ parameterized strategy $\pi_\theta$, we directly sample the gradient and then take the gradient of it, Pay attention we are optimizing the $\epsilon$-greedy policy. and we decrease the $\epsilon$ as progressing further.

#### Mathmatical Calculations

let the benifit function $J(\theta)=\int_\tau p_\theta(\tau)\cdot R_\tau-\lambda\sqrt{\int_\tau p_\theta(\tau)\cdot R_\tau^2-(\int_\tau p_\theta(\tau)\cdot R_\tau)^2}$

$\nabla_\theta J(\theta)=\int_\tau (p_\theta(\tau) \nabla log(p_\theta(\tau))\cdot R_\tau)-$ $\lambda \cdot \frac{ \int_\tau (p_\theta(\tau) \nabla log(p_\theta(\tau))\cdot R_\tau^2) - 2 \int_\tau (p_\theta(\tau) \nabla log(p_\theta(\tau))\cdot R_\tau) \cdot \int_\tau p_\theta(\tau)\cdot R_\tau }{2 \sqrt{\int_\tau p_\theta(\tau)\cdot R_\tau^2-(\int_\tau p_\theta(\tau)\cdot R_\tau)^2}}$

writing into expectation form $\nabla J(\theta)=E(\nabla log(p_\theta(\tau))\cdot R_\tau)-\lambda \frac{E[\nabla log(p_\theta(\tau))\cdot R_\tau^2]-2E[R_\tau] \cdot E[\nabla log(p_\theta(\tau))\cdot R_\tau] }{2\sqrt{E[ R_\tau^2]-E[R_\tau]^2}}$

Thus, we need to sample four quantities in one turn.

#### Mathmatical Interpretation for $\lambda$

- by Chipchoff's inequality $P(|X-E(X)|>\epsilon)<\frac{Var(X)}{\epsilon^2}$

- we take outcome by pessimistic result less than $E(x)-\lambda$ with probalbility at most $\frac{1}{\lambda^2}$ this bound is tight without other information besides second momentum

##### Designed Experiment

###### Aim of experiment

To show that use our algorithm will get a better worst-case reward.

1. A man-made MDP with a high-risk act that averagely benefits more. And an act that gain with a lower risk but gain less reward
2. ...
