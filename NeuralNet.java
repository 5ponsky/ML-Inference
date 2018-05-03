import java.util.ArrayList;
import java.util.Random;

public class NeuralNet extends SupervisedLearner {
  // This number is persistent between epochs
  // It allows for decreasing learning rates
  private double learning_scale = 1.0;
  private double learning_rate = 0.0175;


  private int reg_mode = 2; // temporary place holder for regularization
  private double lambda_1 = 0.008;
  private double lambda_2 = 0.006;

  protected int trainingProgress;

  protected Vec weights;
  protected Vec gradient;
  protected ArrayList<Layer> layers;

  public int[] indices; // Bootstrapping indices

  public Vec cd_gradient;


  String name() { return ""; }

  NeuralNet(Random r) {
    super(r);
    layers = new ArrayList<Layer>();

    trainingProgress = 0;
  }

  void initWeights() {
    // Calculate the total number of weights
    int weightsSize = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      weightsSize += l.getNumberWeights();
    }
    weights = new Vec(weightsSize);
    gradient = new Vec(weightsSize);

    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);

      int weightsChunk = l.getNumberWeights();
      Vec w = new Vec(weights, pos, weightsChunk);

      l.initWeights(w, this.random);

      pos += weightsChunk;
    }
  }

  Vec predict(Vec in) {
    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      int weightsChunk = l.getNumberWeights();
      Vec v = new Vec(weights, pos, weightsChunk);
      l.activate(v, in);
      in = l.activation;
      pos += weightsChunk;
    }

    return (layers.get(layers.size()-1).activation);
  }

  /// Propagate blame from the output side to the input
  void backProp(Vec target) {
    Vec blame = new Vec(target.size());
    blame.add(target);
    blame.addScaled(-1, layers.get(layers.size()-1).activation);

    // keeping this around for good measure?
    layers.get(layers.size()-1).blame = new Vec(blame);

    int pos = weights.size();
    for(int i = layers.size()-1; i >= 0; --i) {
      Layer l = layers.get(i);
      //l.debug();

      int weightsChunk = l.getNumberWeights();
      pos -= weightsChunk;
      Vec w = new Vec(weights, pos, weightsChunk);

      blame = l.backProp(w, blame);
    }
  }

  void updateGradient(Vec x) {

    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      int gradChunk = l.getNumberWeights();
      Vec v = new Vec(gradient, pos, gradChunk);

      l.updateGradient(x, v);
      x = new Vec(l.activation);

      //System.out.println("Layer " + i + ":\n" + gradient);
      pos += gradChunk;
    }
  }

  /// Update the weights
  void refineWeights(double learning_rate) {
    weights.addScaled(learning_rate, gradient);
  }

  /// Trains with a set of scrambled indices to improve efficiency
  void train(Matrix features, Matrix labels, int[] indices, int batch_size, double momentum) {
    if(batch_size < 1)
      throw new IllegalArgumentException("Batch Size < 1");
    if(momentum < 0.0)
      throw new IllegalArgumentException("Momentum < 0");

    // How many patterns/mini-batches should we train on before testing?
    final int cutoff = features.rows();

    Vec in, target;
    // We want to check if we have iterated over all rows
    for(; trainingProgress < features.rows(); ++trainingProgress) {
      in = features.row(indices[trainingProgress]);
      target = labels.row(indices[trainingProgress]);

      predict(in);
      backProp(target);
      updateGradient(in);

      if((trainingProgress + 1) % batch_size == 0) {

        // If we have some form of weight regularization
        if(reg_mode == 1) { // L1 regularization
          l1_regularization();
        } else if(reg_mode == 2) { // L2 regularization
          l2_regularization();
        } else if(reg_mode == 3) { // LP regularization
          lp_regularization(3);
        }
        refineWeights(learning_rate * learning_scale);

        if(momentum <= 0)
          gradient.fill(0.0);
        else
          gradient.scale(momentum);

        // Cut off for intra-training testing
        if(((trainingProgress + 1) / batch_size) % cutoff == 0) {
          ++trainingProgress;
          break;
        }
      }
    }


    // if We have trained over the entire given set
    if(trainingProgress >= features.rows()) {
      trainingProgress = 0;

      // Decrease learning rate
      if(learning_rate > 0)
        learning_scale -= 0.000001;

      scrambleIndices(random, indices, null);
    }
  }

  /// This is an unsupervised learning method designed for images
  void train_with_images(Matrix x) {
    // width and height of the image are referred to as width, height
    int width=0, height= 0; // Place holders

    int channels = x.cols() / (width * height);

    // TODO: initialize the MLP to have channels output units (not x.cols() outputs)

    int k = 2; // # of degrees of freedom in the system
    Matrix states = new Matrix(x.cols(), k);
    states.fill(0.0);

    double learning_rate_local = 0.1;
    for(int j = 0; j < 10; ++j) {
      for(int i = 0; i < 10000000; ++i) {
        int rand_row = random.nextInt(x.rows());
        int p = random.nextInt(width);
        int q = random.nextInt(height);

        Vec features = new Vec(1);
        // TODO: features := a fector containing p/width, q/height, and states[t]

        int s = channels * (width * q + p);

        Vec label;
        // TODO: label:= the vector from X[t][s] to X[t][s + (channels-1)]

        Vec pred = new Vec(predict(features));

        // TODO: compute the error on the output units

        // TODO: do backpropagation to compute the errors of the hidden units

        // TODO: compute the blame terms for V[t]

        // TODO: use gradient descent to refine the weights and bias values

        // TODO: use gradient descent to update V[t]

      }

      learning_rate_local *= 0.75;
    }
  }

  /// L1 regularization
  void l1_regularization() {
    for(int i = 0; i < weights.size(); ++i) {
      double weight = weights.get(i);

      if(weight > 0.0) {
        weights.set(i, weight - (lambda_1 * learning_rate));
      } else if(weight < 0.0) {
        weights.set(i, weight + (lambda_1 * learning_rate));
      }
    }
  }

  void l2_regularization() {
    for(int i = 0; i < weights.size(); ++i) {
      double weight = weights.get(i);

      if(weight > 0.0) {
        weights.set(i, weight - (weight * lambda_2 * learning_rate));
      } else if(weight < 0.0) {
        weights.set(i, weight + (weight * lambda_2 * learning_rate));
      }
    }
  }

  void lp_regularization(int p) {
    int power = p - 1;

    for(int i = 0; i < weights.size(); ++i) {
      double weight = weights.get(i);
      double res = 1;

      // raise a weight to a p-1 power
      for(int j = 0; j < power; ++j) {
        res *= weight;
      }

      if(weight > 0.0) {
        weights.set(i, weight - (res * lambda_1 * learning_rate));
      } else if(weight < 0.0) {
        weights.set(i, weight + (res * lambda_1 * learning_rate));
      }
    }
  }

  void finite_difference(Vec x, Vec target) {
    double h = 1e-6;
    double pred_diff = 1.0;

    /// Calculate gradient with finite difference
    Matrix measured = new Matrix(target.size(), weights.size());
    for(int i = 0; i < weights.size(); ++i) {
      double weight = weights.get(i);

      // Move a weight a little to the left and calculate the output
      weights.set(i, weight + h);
      Vec pred_pos = new Vec(predict(x));

      // Move a weight a little to the right and calculate the output
      weights.set(i, weight - h);
      Vec pred_neg = new Vec(predict(x));

      // put the weight back
      weights.set(i, weight);

      // for each adjusted weight, push the finite difference result into a column
      // each column has the difference for a single adjusted weight
      for(int j = 0; j < target.size(); ++j) {
        double result = (pred_pos.get(j) - pred_neg.get(j)) / (2 * h);
        measured.row(j).set(i, result);
      }
    }

    /// Calulate difference using backprop
    Matrix computed = new Matrix(target.size(), weights.size());
    Vec pred = new Vec(predict(x));
    for(int i = 0; i < target.size(); ++i) {
      double pred_i = pred.get(i);
      pred.set(i, pred_i + pred_diff);
      backProp(pred);
      pred.set(i, pred_i);

      computed.row(i).fill(0.0); // create a gradient
      this.gradient = computed.row(i); // give the NN this gradient row
      updateGradient(x);
    }


    System.out.println("measured:\n" + measured + "\n--------------------------------------");
    System.out.println("computed:\n" + computed + "\n--------------------------------------");

    /// Check results
    int count = 0;
    double sum = 0.0;
    double sum_of_squares = 0.0;
    for(int i = 0; i < target.size(); ++i) {
      for(int j = 0; j < weights.size(); ++j) {
        if(Math.abs(measured.row(i).get(j) - computed.row(i).get(j)) > 1e-5) {

          double err = Math.abs(measured.row(i).get(j) - computed.row(i).get(j));
          throw new RuntimeException("dist(" + measured.row(i).get(j) + ", " + computed.row(i).get(j)
            + ") = " + err + "is too large!");
        } else {
          //System.out.println("match at (i, j): (" + i + ", " + j + ")");
        }

        sum += computed.row(i).get(j);
        sum_of_squares += (computed.row(i).get(j) * computed.row(i).get(j));
      }
    }

    double ex = sum / (target.size() * weights.size());
    double exx = sum_of_squares / (target.size() * weights.size());
    if(Math.sqrt(exx - ex * ex) < 0.01)
      throw new RuntimeException("not enough deviation");

    System.out.println("If the test fails at any point, an exception would have been thrown");
    System.out.println("The printing of this message indicates that the test has passed");
  }

}
