# 📌 Probablistic-Forecasting for Uncertanity Qunatification

## **🔍 Overview**
Tube Loss is a novel loss function designed for **simultaneous estimation** of **Prediction Intervals (PIs)** in regression and probabilistic forecasting tasks. It ensures that the predicted PI achieves the target confidence level **asymptotically** while minimizing its width. Unlike traditional methods, Tube Loss enables **gradient-based optimization**, making it more efficient and scalable for deep learning models.

---

## **📐 Mathematical Definition**
Given a dataset \( \mathcal{D} = \{ (x_i, y_i) \}_{i=1}^{m} \), where \( x_i \in \mathbb{R}^n \) and \( y_i \in \mathbb{R} \), we define a **Prediction Interval (PI)**:

\[
\mu_1(x) \leq y \leq \mu_2(x)
\]

such that:

\[
P(\mu_1(x) \leq y \leq \mu_2(x)) = t
\]

where \( \mu_1(x) \) and \( \mu_2(x) \) are the **lower and upper bounds** of the PI, respectively.

### **📊 Tube Loss Function**
The Tube Loss function is formulated as:

\[
\rho_t^r (y, \mu_1, \mu_2) =
\begin{cases}
t (y - \mu_2), & \text{if } y > \mu_2, \\
(1 - t)(\mu_2 - y), & \text{if } \mu_1 \leq y \leq \mu_2 \text{ and } y \geq r \mu_2 + (1 - r) \mu_1, \\
(1 - t)(y - \mu_1), & \text{if } \mu_1 \leq y \leq \mu_2 \text{ and } y < r \mu_2 + (1 - r) \mu_1, \\
t (\mu_1 - y), & \text{if } y < \mu_1.
\end{cases}
\]

where:
- \( t \) is the **target confidence level**.
- \( r \) is a **user-defined parameter** controlling the PI bounds’ positioning.

---

## **🎯 Key Advantages**
✅ **Simultaneous PI Bound Estimation** → Single optimization problem, unlike quantile regression.  
✅ **Guaranteed Asymptotic Coverage** → Ensures \( PICP \to t \) as sample size increases.  
✅ **Gradient-Based Optimization** → Trainable using **Adam, RMSprop, SGD** (unlike LUBE).  
✅ **Handles Skewed Data** → Adjusts PI placement using parameter \( r \).  
✅ **Minimizes Prediction Interval Width (MPIW)** → Captures the densest region of \( y \).  

---

## **💻 Implementation in Deep Learning**
### **📌 Tube Loss Function in TensorFlow/Keras**
```python
import keras.backend as K

def tube_loss(y_true, y_pred, t=0.95, r=0.5):
    """
    Custom Tube Loss function for Prediction Interval estimation.

    Parameters:
    - y_true: True values
    - y_pred: Predicted [upper bound, lower bound] pairs
    - t: Confidence level
    - r: Shift parameter (default 0.5 for symmetric intervals)

    Returns:
    - Mean Tube Loss value
    """
    y_true = K.squeeze(y_true, axis=-1)  # Remove extra dimension if present
    f1 = y_pred[:, 0]  # Upper bound
    f2 = y_pred[:, 1]  # Lower bound

    loss = K.switch(K.greater(y_true, f1), t * (y_true - f1), 
            K.switch(K.less_equal(y_true, f2), (1 - t) * (f2 - y_true),
            K.switch(K.greater_equal(y_true, r * f2 + (1 - r) * f1), (1 - t) * (f2 - y_true),
            (1 - t) * (y_true - f1))
        )
    )
    
    return K.mean(loss)
