# Machine Learning Crash Course for Physicists in Three Easy Lessons

Written by Florian Marquardt, Max Planck Institute for the Science of Light, Germany (originally developed for a Trieste Summer School in 2024).

This online book teaches machine learning to a physicist in three easy chapters. You should be able to work through them in three to four hours total, provided you know how to use python and have worked with numpy before. You can run the code examples in your browser, without any installation, by clicking on the rocket (taking you to Google colab).

This book uses the [jax](https://jax.readthedocs.io) machine learning library, which is really easy to use, since it is directly based on numpy and only adds a handful simple but powerful commands. jax is also state of the art, being used in famous recent projects like DeepMind's AlphaFold. Afterwards, you will already be able to use neural networks to train them on experimental data or simulations, for tasks like rapid experimental calibration or speed-up of simulations, and many more.

- The first chapter shows how to minimize the energy of a mechanical structure, relaxing it to equilibrium. This teaches how to use automatic gradients and adaptive-stepsize gradient descent.

- The second chapter shows how to approximate an arbitrary function using a simple neural network. Training of that network uses the concepts already introduced in chapter 1. In addition, we learn about batch processing and about random numbers in jax.

- The third and final chapter shows how to apply this to an interesting physics case. Suppose you take data in an experiment where you extract the response of a driven nonlinear oscillator. Your goal is to guess the correct underlying system parameters (like resonance frequency and damping) rapidly. A neural network, once trained, can do that for you. It will be much faster and more robust than usual nonlinear curve fitting.

You can view the slides that went along with the in-person lectures as a [pdf](https://github.com/FlorianMarquardt/MachineLearningThreeEasyLessons/blob/main/2024_09_ML_Three_Easy_Lessons_Marquardt.pdf) -- best open this in a separate tab.

