# membership-inference-attack
This code demonstrates a Model Inference Attack (MIA) against a digit recognition model trained on MNIST dataset.

has the goal to determine whether specific sample (in this case, a handwritten digit) was part of the target model's training dataset, which indicates a serious privacy concern in sensitive apps (imagine it was determining whether a specific person's photo was used in facial recognition model instead).

corresponding defense mech using regularization and dropout are implemented to mitigate this attack.