1) 首先，我们要求的是$\frac{\partial \sigma_p}{\partial w_i}$，这是一个关于$w_i$的偏导数。
   虽然我们知道$\sigma_p = 13.95\%$，但为了求导，我们需要知道$\sigma_p$是如何由$w_i$构成的。

2) 我们从$\sigma_p = \sqrt{w^T\Sigma w}$开始。使用链式法则：
   - 令$u = w^T\Sigma w$，则$\sigma_p = \sqrt{u}$
   - $\frac{\partial \sqrt{u}}{\partial w_i} = \frac{1}{2\sqrt{u}} \cdot \frac{\partial u}{\partial w_i}$


3) 当我们展开$u = w^T\Sigma w = \sum_{j=1}^n\sum_{k=1}^n w_j w_k \sigma_{jk}$时，要找出所有含$w_i$的项。
   由于$w_i$可能是第一个乘数($w_j$)或第二个乘数($w_k$)，我们需要分别考虑：

   a. 当$w_i$是第一个乘数时（即j = i）：
      - 这种情况下形成的项是：$w_i(w_1\sigma_{i1} + w_2\sigma_{i2} + ... + w_i\sigma_{ii} + ... + w_n\sigma_{in})$
      - 用求和符号表示就是：$w_i\sum_{k=1}^n w_k \sigma_{ik}$

   b. 当$w_i$是第二个乘数时（即k = i）：
      - 这种情况下形成的项是：$w_i(w_1\sigma_{1i} + w_2\sigma_{2i} + ... + w_i\sigma_{ii} + ... + w_n\sigma_{ni})$
      - 用求和符号表示就是：$w_i\sum_{j=1}^n w_j \sigma_{ji}$

4) 现在对这两类项分别求导：

   a. 对第一类项$w_i\sum_{k=1}^n w_k \sigma_{ik}$求导：
      - 这是一个乘积，要用乘积法则$\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$
      - 其中$f(w_i) = w_i$，$g(w_i) = \sum_{k=1}^n w_k \sigma_{ik}$
      - 前导后不导：$f'(w_i) \cdot g(w_i) = 1 \cdot (w_1\sigma_{i1} + w_2\sigma_{i2} + ... + w_n\sigma_{in})$
      - 前不导后导：$f(w_i) \cdot g'(w_i) = w_i \cdot \sigma_{ii}$（因为求和式中只有$w_i\sigma_{ii}$项对$w_i$求导不为0）
      - 加起来得到：$(w_1\sigma_{i1} + w_2\sigma_{i2} + ... + w_n\sigma_{in}) + w_i\sigma_{ii}$
      - 这就等于$\sum_{k=1}^n w_k \sigma_{ik}$，也就是$(\Sigma w)_i$

   b. 对第二类项进行同样的求导过程，由于协方差矩阵对称（$\sigma_{ij} = \sigma_{ji}$），
      得到相同的结果：$\sum_{j=1}^n w_j \sigma_{ji} = \sum_{k=1}^n w_k \sigma_{ik} = (\Sigma w)_i​$

   因此，总的求导结果是这两个相等的项之和：$2(\Sigma w)_i$

5) 对第二类项求导，由于协方差矩阵对称（$\sigma_{ij} = \sigma_{ji}$），得到相同的结果。

6) 因此$\frac{\partial u}{\partial w_i} = 2(\Sigma w)_i$

7) 代回链式法则：
   $$\frac{\partial \sigma_p}{\partial w_i} = \frac{1}{2\sqrt{u}} \cdot 2(\Sigma w)_i = \frac{(\Sigma w)_i}{\sqrt{w^T\Sigma w}} = \frac{(\Sigma w)_i}{\sigma_p}$$

8) 对于NVDA：
   - $(\Sigma w)_{NVDA} = 0.049829$（来自矩阵乘法）
   - $\sigma_p = 0.1395$（已知的组合标准差）
   - 所以$MRC_{NVDA} = 0.049829/0.1395 = 0.3572$

这个推导过程展示了为什么最终的MRC公式如此简洁 - 所有的复杂性都被乘积法则和链式法则的运算化简了。

这就是为什么最终公式变得这么简单的原因 - 所有的复杂运算都在推导过程中相互抵消了。
