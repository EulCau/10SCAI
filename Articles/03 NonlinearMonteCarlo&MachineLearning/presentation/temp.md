# 翻译

## 4 高维控制问题

高维问题常在最优控制中出现。事实上，“维数灾难”这一概念最早就是在控制问题的动态规划背景下，由理查德·贝尔曼提出的 [13]。关于维数灾难，不同于开环控制和闭环控制之间存在一个非常重要的区别，下面我们对此进行说明。

考虑具有有限时域 $T$ 的最优控制问题：

$$
\min_{u} \quad g(x(T)) + \int_{0}^{T} L(t, x(t), u(t))\,dt,
$$

受限于动态约束

$$
\dot{x}(t) = f(t, x(t), u(t)), \quad x(0)=x_{0}. \quad (49)
$$

其中，状态 $x:[0,T] \to X\subset\mathbb{R}^d$ ，控制 $u:[0,T] \to U\subset\mathbb{R}^m$ ，终端代价 $g:X\to\mathbb{R}$ 以及运行代价 $L:[0,T]\times X\times U\to\mathbb{R}$ 均已给定。对于固定的初始状态 $x_0$ ，上述问题可以看作是定义在时间区间 $[0,T]$ 上的一个两点边值问题，而最优控制往往以如下形式给出：

$$
u = u^*(t, x^*(t)), \quad (50)
$$

其中 $x^*(t)$ 表示最优轨迹。

在这种情形下，由于自变量（即时间）的维数仅为1，维数灾难并不是主要问题，因此这种形式的控制被称为开环控制。开环控制下，最优控制仅在最优轨迹上有效；一旦系统偏离了最优轨迹，就需要重新计算最优控制或者将系统强制回归到最优轨迹。然而，在许多实际应用中，人们更倾向于使用闭环控制（或反馈控制），即设计形式为

$$
u = u^*(t, x),
$$

使得在状态空间中的每一点都能确定最优控制。闭环控制是状态变量的函数，而这正是维数灾难问题出现的根源所在。为了对开环控制与闭环控制进行刻画，令扩展哈密顿量为

$$
\tilde{H}(t,x,\lambda,u)=L(t,x,u)+\lambda^T f(t,x,u), \quad (52)
$$

并定义

$$
u^*(t,x;\lambda)=\arg\min_{u\in U} \tilde{H}(t,x,\lambda,u), \quad (53)
$$

其中 $\lambda$ 为伴随变量。一个重要的结论是，最优控制问题的解满足庞特里亚金极小值原理，即存在满足下列条件的解：

$$
\begin{cases}
\dot{x}(t)=\frac{\partial \tilde{H}}{\partial \lambda}(t,x,\lambda,u^*(t,x,\lambda))=f(t,x,u^*(t,x,\lambda)),\\[1mm]
\dot{\lambda}(t)=-\frac{\partial \tilde{H}}{\partial x}(t,x,\lambda,u^*(t,x,\lambda)),\\[1mm]
\dot{v}(t)=-L(t,x,u^*(t,x,\lambda)),
\end{cases}
\quad (54)
$$

并满足边界条件

$$
x(0)=x_0,\quad \lambda(T)=\nabla g(x(T)),\quad v(T)=g(x(T)). \quad (55)
$$

记控制问题的值函数为

$$
V(t,x)=\inf_{u\in U} \{ g(y(T))+\int_{t}^{T} L(\tau,y,u)d\tau \},
$$

其中状态轨迹 $y(\tau)$ 满足动态约束

$$
\dot{y}(\tau)=f(\tau,y,u) \quad \text{且} \quad y(t)=x.
$$

定义哈密顿量

$$
H^*(t,x,\lambda)=\tilde{H}(t,x,\lambda,u^*(t,x,\lambda)). \quad (57)
$$

由此，Hamilton-Jacobi-Bellman (HJB) 方程可以写成

$$
V_t(t,x)+H^*(t,x,V_x)=0, \quad (58)
$$

并附有终端条件 $V(T,x)=g(x)$。利用值函数，可以表达伴随变量与闭环最优控制为

$$
\lambda(t)=\nabla_x V(t,x(t)), \quad (59)
$$

$$
u^*(t,x)=\arg\min_{u\in U}H(t,x,\nabla_x V,u). \quad (60)
$$

为了获得闭环控制的精确逼近，必须对大范围甚至全状态空间内的初始条件求解控制问题；而前述公式 (49) 仅针对单一初始条件。为推广到所有初始条件，我们可以考虑如下问题：

$$
\min_{u} \; E_{x_0\sim \mu}\Big[ g(x(T))+\int_{0}^{T}L(t,x(t),u(t,x(t)))\,dt \Big], \quad (61)
$$

其受限于动态系统

$$
\dot{x}(t)=f(t,x(t),u(t,x(t)))\quad \text{且} \quad x(0)=x_0.
$$

这里，优化变量为所有可能的策略函数 $u$。一个自然的问题是如何选择初始状态的分布 $\mu$？显然，我们仅对那些值函数较小的状态感兴趣，因此一种可能的选择是采用值函数的 Gibbs 分布，即

$$
\mu=\frac{1}{Z}e^{-\beta V}, \quad (62)
$$

其中 $Z$ 是归一化因子，$\beta>0$ 是超参数。

需要指出的是，与随机情形下训练数据可以“现成”在线生成不同，此处必须明确解决数据生成的问题。为此，在文献 [117,97] 中提出了以下策略：

- 首先，通过求解两点边值问题 (54)–(55) 来生成训练数据；
- 其次，利用生成的数据训练神经网络模型以逼近值函数。

实际上，求解 (54)–(55) 本身并非易事，因此寻找一个既小又具有代表性的训练数据集显得尤为重要。为此，文献 [117,97] 中探讨了两种方法：

1. “热启动”策略。其基本思想是，为 (54)–(55) 的迭代算法选择合适的初始化，从而有助于保证算法收敛。例如，可以先从较短的时间区间 $T$ 开始，此时迭代收敛性问题较轻，然后将得到的解简单外推到更长的时间区间，作为后续求解的初值，如此迭代进行；
2. 自适应采样。类似的思想也在 [147] 中有所探讨。对于自适应算法而言，关键在于设计误差指示器：误差较大时，需要采集更多数据。文献 [147] 中采用来自多个相似机器学习模型预测值的方差作为误差指示器；而 [117] 则提出了一种基于损失函数梯度方差的精细误差指示方法，另有观点认为可以直接使用值函数梯度的幅值作为误差指标。

---

## 5 Ritz、Galerkin 和最小二乘法

Ritz 公式、Galerkin 公式以及最小二乘法是设计求解偏微分方程（PDE）数值算法中常用的几种框架。其中，Ritz 公式基于变分原理；Galerkin 公式则基于 PDE 的弱形式，涉及试探函数与测试函数；而最小二乘法是一种将 PDE 问题转化为变分问题的方法，其基本思路是最小化 PDE 残差的平方。最小二乘法具有通用性和直观性，但在传统数值分析中往往不被优先选择，因为由此得到的数值问题通常比 Ritz 或 Galerkin 公式的条件更差。利用变分原理构造基于机器学习的算法（例如将试探函数空间替换为机器学习模型的假设空间）较为直接，而采用 Galerkin 公式则不同，它依赖于弱形式和测试函数。实际上，与 Galerkin 公式最为接近的机器学习模型是 Wasserstein GAN（WGAN）：在 WGAN 中，判别器承担了测试函数的角色，而生成器则相当于试探函数。

## 5.1 Deep Ritz 方法

Deep Ritz 方法最早在文献 [41] 中提出。考虑如下变分问题 [43]：

$$
\min_{u \in H} I(u) \quad \text{(63)}
$$

其中，

$$
I(u) = \int_{\Omega} \left(\frac{1}{2} |\nabla u(x)|^2 - f(x) u(x)\right) dx \quad \text{(64)}
$$

这里 $H$ 表示一组允许的函数（即试探函数，此处用 $u$ 表示），$f$ 为给定的函数，用以描述系统受到的外部激励，同时边界条件也已在 $H$ 的定义中包含。Deep Ritz 方法主要包含以下几个组成部分：

1. 使用深度神经网络表示试探函数；
2. 针对泛函 $I(u)$ 采用数值积分（即利用数值求积公式）进行离散化；
3. 设计求解最终优化问题的算法。

各个组件较为直观。在高维情形下，需要借助高效的 Monte Carlo 算法来离散化泛函 (64) 中的积分。需要注意的是，积分的离散化与利用神经网络表示试探函数之间的相互作用是一个值得深入研究的问题。最后，由于泛函 (64) 类似于 Deep BSDE 中期望的表达，因此可自然地采用随机梯度下降（SGD）进行优化。在激活函数的选择上存在一个显著问题：由于 ReLU 的导数存在不连续性，其表现往往不如平滑版本。有研究表明激活函数 $\sigma_3(z)=\max(z,0)$ 的表现优于传统的 ReLU，关于这一问题仍需更深入探讨。值得指出的是，Deep Ritz 方法的一个潜在优势在于它是无网格且自适应的。为验证这一点，研究者们考察了著名的裂纹问题——即计算裂纹附近的位移。为此，考虑 Poisson 方程：

$$
-\Delta u(x) = 1,\quad x \in \Omega, \qquad u(x) = 0,\quad x \in \partial\Omega \quad \text{(65)}
$$

其中 $\Omega = (-1,1)\times(-1,1)\setminus ([0,1)\times\{0\})$，该问题的解由于域的几何形状而呈现出著名的“角点奇异性”。简单的渐进分析表明，在原点附近，解满足
$$
u(x)=u(r,\theta) \sim r^{\frac{1}{2}} \sin^2\theta
$$
的行为 [134]。

在传统有限元方法中，此类模型被广泛用于开发和测试自适应方法，而在处理本质边界条件时会遇到一些问题。最简单的思路是采用罚函数方法，对泛函做如下修正：

$$
I(u) = \int_{\Omega} \left(\frac{1}{2} |\nabla u(x)|^2 - f(x) u(x)\right) dx + \beta \int_{\partial\Omega} u(x)^2 \, ds \quad \text{(66)}
$$

通常可选 $\beta=500$。文献 [41] 中给出了一个具体示例，其采用 811 个参数的神经网络模型求得的 Deep Ritz 方法解，与采用间距 $\Delta x_1=\Delta x_2=0.1$（自由度为 1681）的有限差分法结果对比，效果良好。更为量化的比较可参见 [41]。当然，自适应数值方法在求解具有角点奇异性甚至更一般的奇异问题上已有成熟的发展，然而此例说明 Deep Ritz 方法在自适应性方面具有一定优势。未来的研究还需解决以下问题：

1. 尽管原问题可能是凸的，但由 Deep Ritz 得到的变分问题通常并非凸；
2. 目前尚未就其收敛速度达成一致结论；
3. 本质边界条件的处理没有传统方法那样简单。

关于 Deep Ritz 方法的一些理论分析可参见文献 [116]。

## 5.2 最小二乘法公式

最小二乘法方法最初在文献 [22] 中用于求解动态 Schrödinger 方程，随后在 [132] 中被更系统地发展（虽然 [132] 将其称为 Galerkin 方法）。其基本思路十分简单：考虑在区域 $\Omega\subset \mathbb{R}^d$ 上求解 PDE

$$
Lu = f \quad \text{(67)}
$$

这一问题可以等价地转化为下面的变分问题，即求解泛函

$$
J(u) = \int_{\Omega} \|Lu - f\|^2 \, \mu(dx) \quad \text{(68)}
$$

的最小值，其中 $\mu$ 是在 $\Omega$ 上选取的一个适当的概率分布。要求 $\mu$ 非退化且易于采样。从形式上看，最小二乘法的变分问题与 Ritz 方法中的泛函 $I(u)$ 类似，只是此处用 $J(u)$ 替换了 $I(u)$。

## 5.3 Galerkin 公式

Galerkin 方法的出发点是 PDE (67) 的弱形式，即：寻找 $u\in H_1$，使得对于任意测试函数 $\varphi\in H_2$ 有

$$
a(u, \varphi) = (Lu, \varphi) = (f, \varphi) \quad \text{(69)}
$$

其中 $H_1$ 与 $H_2$ 分别为试探函数空间和测试函数空间，$\varphi$ 为 $H_2$ 中任一函数，$(\cdot,\cdot)$ 表示 $L^2$ 内积。通常情况下，为了降低对高阶导数的需求，会对积分项进行分部积分。例如，当 $L=-\Delta$ 时（忽略边界项），有

$$
a(u, \varphi) = (\nabla u, \nabla \varphi) \quad \text{(70)}
$$

因此，这种弱形式仅涉及一阶导数。Galerkin 方法的一个重要特点在于它同时引入了测试函数。基于这一点，Wasserstein GAN（WGAN）可以看作是一种 Galerkin 近似方法：给定一组数据 $\{x_j,\; j=1,2,\ldots,n\}$ 以及一个参考概率分布 $\nu^*$ 在 $\mathbb{R}^{d'}$ 上，我们寻求一个映射 $G:\mathbb{R}^{d'}\to\mathbb{R}^d$，使得对于所有满足 Lipschitz 条件的函数 $\varphi$ 有

$$
\int_{\mathbb{R}^{d'}} \varphi(G(z)) \, \nu^*(dz) = \frac{1}{n}\sum_{j=1}^{n} \varphi(x_j) \quad \text{(71)}
$$

在这种思路下，最直观的将 (69) 改写为一个极小极大问题为

$$
\min_{u\in H_1} \max_{\|\varphi\|_{H_2} \leq 1} \Big(a(u,\varphi) - (f,\varphi)\Big)^2 \quad \text{(72)}
$$

但遗憾的是，这种形式的公式较难直接求解，其遇到的问题与 WGAN 中类似。尽管如此，一些令人鼓舞的进展已经取得，有关具体细节可参考文献 [143]。

---

## 第六章 用于非线性偏微分方程的多层 Picard 近似方法

在文章 [37]（E 等人）和 [89]（Hutzenthaler 等人）中，提出并分析了所谓的全历史递归多层 Picard 近似方法（以下简称 MLP 方法）。在 [89] 中，对半线性热偏微分方程（具有 Lipschitz 非线性）的误差分析仅限于这一类问题。然而，在科学文献中，现已有一系列关于 MLP 近似方法的进一步文章（参见 [90, 7, 53, 6, 9, 86, 91, 38, 87]），这些文章分析、扩展或推广了在 [37, 89] 中提出的 MLP 近似方法，使其适用于更大类的偏微分方程问题，例如半线性 Black-Scholes 偏微分方程（参见 [90, 9]）、具有梯度依赖非线性项的半线性热偏微分方程（参见 [86, 91]）、半线性椭圆偏微分方程问题（参见 [6]）、具有非 Lipschitz 连续非线性项的半线性热偏微分方程（参见 [7, 9]），以及系数函数变化的半线性二阶偏微分方程（参见 [90, 87]）。

在本章余下部分中，我们将大致描述 MLP 近似方法的主要思想。为使表述尽可能简明易懂，下面的讨论将限制在具有有界初值且非线性项为 Lipschitz 连续、仅依赖于偏微分方程解本身的半线性热偏微分方程情形下。接下来的结果，即定理 3，为在上述条件下使用 MLP 近似方法求解半线性热偏微分方程提供了复杂度分析。定理 3 的证明主要基于 Hutzenthaler 等人 [89, Theorem 1.1] 与 Beck 等人 [7, Theorem 1.1] 的工作。

>### 定理 3
>
>设 $T \in (0,\infty)$，令
>
>$$
>\Theta = \bigcup_{n \in \mathbb{N}} \mathbb{Z}^n,
>$$
>
>设 $f: \mathbb{R} \to \mathbb{R}$ 为 Lipschitz 连续函数，对于每个 $d \in \mathbb{N}$，令
>
>$$
>u_d \in C^{1,2}([0,T] \times \mathbb{R}^d, \mathbb{R})
>$$
>
>满足多项式增长条件，并且对所有 $d \in \mathbb{N}$、$t \in [0,T]$ 及 $x \in \mathbb{R}^d$ 有
>
>$$
>\frac{\partial}{\partial t} u_d(t,x) = \Delta_x u_d(t,x) + f(u_d(t,x)).
>$$
>
>设 $(\Omega, \mathcal{F}, \mathbb{P})$ 为概率空间，对于每个 $\theta \in \Theta$，令
>
>$$
>R_{\theta} : \Omega \to [0,1]
>$$
>
>为独立的 $\mathrm{U}[0,1]$ 分布随机变量，且对于每个 $d \in \mathbb{N}$ 和 $\theta \in \Theta$，令
>
>$$
>W^{d,\theta} : [0,T] \times \Omega \to \mathbb{R}^d
>$$
>
>为独立的标准布朗运动，并假设 $\{ R_{\theta} \}_{\theta \in \Theta}$ 与 $\{ W^{d,\theta} \}_{(d,\theta) \in \mathbb{N} \times \Theta}$ 相互独立。对于每个 $d \in \mathbb{N}$、$s \in [0,T]$、$t \in [s,T]$、$x \in \mathbb{R}^d$ 及 $\theta \in \Theta$，令 $X^{d,\theta}_{s,t,x} : \Omega \to \mathbb{R}^d$ 定义为
>
>$$
>X^{d,\theta}_{s,t,x} = x + \sqrt{2}\, \bigl( W^{d,\theta}_t - W^{d,\theta}_s \bigr).
>$$
>
>设对于每个 $d, M \in \mathbb{N}$、$n \in \mathbb{N}_0$ 及 $\theta \in \Theta$，有随机函数
>
>$$
>U^{d,\theta}_{n,M} : [0,T] \times \mathbb{R}^d \times \Omega \to \mathbb{R},
>$$
>
>满足对于所有 $d, M \in \mathbb{N}$、$n \in \mathbb{N}_0$、$\theta \in \Theta$、$t \in [0,T]$ 和 $x \in \mathbb{R}^d$ 有
>
>$$
>\begin{aligned}
>U^{d,\theta}_{n,M}(t,x) = {} & \sum_{k=1}^{n-1} \Biggl[ \frac{1}{M^{n-k}} \sum_{m=1}^{M^{n-k}} \Bigl( f\Bigl( U^{d,(\theta,k,m)}_{k,M}\bigl(tR_{(\theta,k,m)}, X^{d,(\theta,k,m)}_{tR_{(\theta,k,m)},t,x}\bigr) \Bigr) \\
>& \quad - f\Bigl( U^{d,(\theta,-k,m)}_{k-1,M}\bigl(tR_{(\theta,k,m)}, X^{d,(\theta,k,m)}_{tR_{(\theta,k,m)},t,x}\bigr) \Bigr) \Bigr) \Biggr] \\
>& \quad + \frac{1}{M^n} \sum_{m=1}^{M^n} \Bigl( u_d\bigl(0, X^{d,(\theta,0,-m)}_{0,t,x} \bigr) + t\, f(0) \Bigr).
>\end{aligned}
>$$
>
>（公式中的下标和上标表示对独立随机变量的不同取值，具体定义参见 [87, Corollary 4.4]。）
>
>对于每个 $d, M \in \mathbb{N}$ 和 $n \in \mathbb{N}_0$，令 $C^{d}_{n,M} \in \mathbb{N}_0$ 表示计算 $U^{d,0}_{n,M}(T,0) : \Omega \to \mathbb{R}$ 时所需要的 $f$ 和 $u_d(0,\cdot)$ 的函数值的计算次数以及标量随机变量生成的次数。则存在一个函数
>
>$$
>N: (0,1] \to \mathbb{N}
>$$
>
>和常数 $c \in \mathbb{R}$，使得对于所有 $d \in \mathbb{N}$ 和 $\varepsilon \in (0,1]$ 有
>
>$$
>C^{d}_{N_{\varepsilon},N_{\varepsilon}} \le c\, d\, \varepsilon^{-3}
>$$
>
>且
>
>$$
>\Bigl( \mathbb{E}\Bigl[ \bigl| U^{d,0}_{N_{\varepsilon},N_{\varepsilon}}(T,0) - u_d(T,0) \bigr|^2 \Bigr] \Bigr)^{\frac{1}{2}} \le \varepsilon.
>$$

接下来，我们对定理 3 中所涉及的符号和定义做一些说明：

- 定理 3 中描述的 MLP 近似旨在逼近偏微分方程
  
  $$
  \frac{\partial}{\partial t} u_d(t,x) = \Delta_x u_d(t,x) + f(u_d(t,x))
  $$

  的精确解，其中 $T \in (0,\infty)$ 表示时间终点。
  
- 函数 $f: \mathbb{R} \to \mathbb{R}$ 描述了偏微分方程中的非线性项，此处为简化讨论，我们假设 $f$ 仅依赖于 $u_d(t,x)$，而不依赖于时间 $t$、空间变量 $x$ 或偏微分方程解的导数。

- 对于更一般的情况（例如非线性项可以依赖于 $t$、$x$ 以及 $u_d$ 的导数），相关的 MLP 分析见文献 [86] 和 [7, 9]。

- 函数 $u_d : [0,T] \times \mathbb{R}^d \to \mathbb{R}$ 表示偏微分方程的精确解。

- MLP 近似方法是一种非线性 Monte Carlo 算法，其主要思想是通过引入大量独立同分布的随机变量（利用索引集 $\Theta$）来构造多层递归的 Picard 迭代，从而在一定条件下克服维数灾难。

- 指标 $C^{d}_{n,M}$ 用来衡量在计算 $U^{d,0}_{n,M}(T,0)$ 时所需的计算资源（包括函数计算次数和随机变量生成次数）。定理 3 表明，当所要求的近似精度为 $\varepsilon$ 时，其计算成本最多以 $d$ 和 $\varepsilon^{-3}$ 的多项式级别增长。

文献中还进一步讨论了在更一般条件下，MLP 近似方法如何适用于更广泛的偏微分方程问题，并给出了更加精确的误差常数和参数依赖的指数。与传统的 Monte Carlo 方法相比，MLP 方法在一定条件下能够有效地克服维数灾难，其计算复杂度仅呈多项式增长。
