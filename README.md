# membership-inference-attack
This code demonstrates a Model Inference Attack (MIA) against a digit recognition model trained on MNIST dataset.

has the goal to determine whether specific sample (in this case, a handwritten digit) was part of the target model's training dataset, which indicates a serious privacy concern in sensitive apps (imagine it was determining whether a specific person's photo was used in facial recognition model instead).

A corresponding defense mechanism using regularization and dropout are implemented to mitigate this attack.

**Project Overview**
- Objective: Determine whether specific data points were used in training a target CNN (LeNet).
- Attacker's abilities:
    - Query the target model to observe predictions.
    - Use publicly available datasets (MNIST) to build shadow models.
    - Train an attack model using member vs non-member responses.

**Attack Overview**
- Target Model Training
    - LeNet CNN trained on MNIST.
    - Overfitting encouraged in some runs by using small training set (≤1).
- Shadow Models
    - Multiple shadow models trained on partitioned datasets to approximate the target model's behaviour.
- Attack dataset generation
    - Query shadow models with member & non-member samples.
    - collect predictions to build an attack dataset.

- Attack model training
    - binary classifier trained to distinguissh member vs non-member samples
    - evaluation with accuracy, recall, F1-score and computation time.

**Results**
Overfitted Target Model (Vulnerable)
| Run | Epochs | Accuracy (%) | Precision | Recall | F1-Score | Time (s) |
| --- | ------ | ------------ | --------- | ------ | -------- | -------- |
| 1   | 20     | 67.50        | 0.68      | 0.65   | 0.67     | 3.70     |
| 2   | 25     | 73.68        | 0.70      | 0.84   | 0.76     | 3.84     |
| 3   | 50     | **91.67**    | 0.86      | 1.00   | 0.92     | 3.91     |
| 4   | 100    | 90.35        | 0.93      | 0.88   | 0.90     | 4.14     |

1. Attack is highly succesfull on overfitted models, which peaks at 91.67%
2. Execution (cost computation) is efficient, ~4 secs.

Well-Trained Target Model (Defended)
| Run | Epochs | Accuracy (%) | Precision | Recall | F1-Score | Time (s) |
| --- | ------ | ------------ | --------- | ------ | -------- | -------- |
| 1   | 20     | 50.03        | 0.50      | 0.95   | 0.66     | 19.45    |
| 2   | 25     | 50.01        | 0.50      | 1.00   | 0.67     | 25.64    |
| 3   | 50     | 50.26        | 0.51      | 0.14   | 0.22     | 42.18    |
| 4   | 100    | 51.17        | 0.51      | 0.60   | 0.55     | 81.49    |

1. Defense: Dropout(0.1 - 0.5)
2. With defense mechanism, attack accuracy drops into ~50% (random guessing).


**Defense Mechanism**
1. Regularization (L2-Norm)
    - Penalizes large weights -> promoting generalization.
2. Dropout
    - Randomly drops neurons during train -> prevent overfitting.
3. Balanced training size
    - Larger train_size(≥0.5) ensures the model generalizes better.
4. Evaluatio
    - Defense judged by how close attack accuracy only is to 50%.


**Findings**
1. **Overfitted** Models are highly vulnerable to MIA attacks.
2. **Well trained and regularization models** resist inference attack, keeping the accuracy near random guessing.
3. Defense mechanism adds minimal computatioal overhead, making it practical for real-world prodduction deployments/productions.

**This works is based on paper:** https://arxiv.org/pdf/1610.05820.pdf