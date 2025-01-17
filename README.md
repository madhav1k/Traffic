# Traffic
 Traffic signs recognition AI model. Uses German Traffic Sign Recognition Benchmark (GTSRB) dataset.

- AI to identify which traffic sign appears in a photograph built with Python, OpenCV, TensorFlow, Keras (TensorFlow), Scikit-learn, NumPy, and more.

- Used OpenCV to read & resize images from GTSRB (German Traffic Sign Recognition Benchmark) dataset before feeding it to CNN (Convolutional Neural Network) for training.

- Used Scikit-learn for splitting dataset conveniently for training & testing.

<img width="1200" alt="image" src="https://github.com/user-attachments/assets/e3d024ba-015a-4d19-8658-9a9cdfa6f128" />

## GTSRB

I initially replicated a model with the following architecture: a 2D convolutional layer with 32 filters, a max-pooling layer, a flattening layer, a dense layer with 128 neurons, and a dropout layer at 50%. This resulted in only 5.3% accuracy. Reducing dropout to 30% dramatically improved accuracy to 92%, with further reductions leading to 94.6% accuracy at 5% dropout. However, overfitting occurred at this point.

I then adjusted the number of neurons, increasing from 128 to 256, which improved accuracy slightly but doubled the computation time. Experiments with two dense layers (128 neurons each) reduced performance to 82.5%.

Next, a second convolutional and max-pooling layer was added, boosting accuracy to 97.6%. Using 64 filters instead of 32 filters marginally increased accuracy to 97.9%, but computation time again doubled.

The final model, with 128 neurons, two convolutional layers (32 filters each), two max-pooling layers, and a 10% dropout rate, yielded similar performance to a more complex version with 64 filters and 256 neurons but with much faster training.

<img width="1200" alt="image" src="https://github.com/user-attachments/assets/45d0026f-8ce9-4d44-bf40-2c49ebce789b" />
