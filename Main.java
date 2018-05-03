// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
import java.util.Random;
import java.lang.*;

class Main
{
	static void test(SupervisedLearner learner, String challenge) {
		// Load the training data
		String fn = "data/" + challenge;
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF(fn + "_train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF(fn + "_train_lab.arff");

		// Train the model
		//learner.train(trainFeatures, trainLabels, Training.NONE);

		// Load the test data
		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF(fn + "_test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF(fn + "_test_lab.arff");

		// Measure and report accuracy
		int misclassifications = learner.countMisclassifications(testFeatures, testLabels);
		System.out.println("Misclassifications by " + learner.name() + " at " + challenge + " = " + Integer.toString(misclassifications) + "/" + Integer.toString(testFeatures.rows()));
	}

	public static void testCV(SupervisedLearner learner) {
		Matrix f = new Matrix();
		f.newColumns(1);
		double[] f1 = {0};
		double[] f2 = {0};
		double[] f3 = {0};
		f.takeRow(f1);
		f.takeRow(f2);
		f.takeRow(f3);

		Matrix l = new Matrix();
		l.newColumns(1);
		double[] l1 = {2};
		double[] l2 = {4};
		double[] l3 = {6};
		l.takeRow(l1);
		l.takeRow(l2);
		l.takeRow(l3);

		double rmse = learner.cross_validation(1, 3, f, l, learner);
		System.out.println("RMSE: " + rmse);
	}

	public static void testOLS() {
		LayerLinear ll = new LayerLinear(13, 1);
		Random random = new Random(123456);
		Vec weights = new Vec(14);

		for(int i = 0; i < 14; ++i) {
			weights.set(i, random.nextGaussian());
		}

		Matrix x = new Matrix();
		x.newColumns(13);
		for(int i = 0; i < 100; ++i) {
			double[] temp = new double[13];
			for(int j = 0; j < 13; ++j) {
				temp[j] = random.nextGaussian();
			}
			x.takeRow(temp);
		}

		Matrix y = new Matrix(100, 1);
		for(int i = 0; i < y.rows(); ++i) {
			ll.activate(weights, x.row(i));
			for(int j = 0; j < ll.activation.size(); ++j) {
				double temp = ll.activation.get(j) + random.nextGaussian();
				y.row(i).set(j, temp);
			}
		}

		for(int i = 0; i < weights.size(); ++i) {
    	System.out.println(weights.get(i));
		}

		Vec olsWeights = new Vec(14);
		ll.ordinary_least_squares(x,y,olsWeights);

		System.out.println("-----------------------------");

		for(int i = 0; i < olsWeights.size(); ++i) {
			System.out.println(olsWeights.get(i));
		}
	}


	public static void testLayer() {
		double[] x = {0, 1, 2};
		double[] m = {1, 5, 1, 2, 3, 2, 1, 0};
		LayerLinear ll = new LayerLinear(3, 2);
		ll.activate(new Vec(m), new Vec(x));
		System.out.println(ll.activation.toString());
	}


	public static void opticalCharacterRecognition() {
		Random random = new Random(123456); // used for shuffling data


		/// Load training and testing data
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF("data/train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF("data/train_lab.arff");

		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF("data/test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF("data/test_lab.arff");

		/// Normalize our training/testing data by dividing by 256.0
		/// There are 256 possible values for any given entry
		trainFeatures.scale((1 / 256.0));
		testFeatures.scale((1 / 256.0));

		/// Build index arrays to shuffle training and testing data
		int[] trainingIndices = new int[trainFeatures.rows()];
		int[] testIndices = new int[testFeatures.rows()];

		// populate the index arrays with indices
		for(int i = 0; i < trainingIndices.length; ++i) { trainingIndices[i] = i; }
		for(int i = 0; i < testIndices.length; ++i) { testIndices[i] = i; }

		/// Assemble and initialize a neural net
		NeuralNet nn = new NeuralNet(random);

		nn.layers.add(new LayerLinear(784, 80));
		nn.layers.add(new LayerTanh(80));

		nn.layers.add(new LayerLinear(80, 30));
		nn.layers.add(new LayerTanh(30));

		nn.layers.add(new LayerLinear(30, 10));
		nn.layers.add(new LayerTanh(10));

		nn.initWeights();


		/// Training and testing
		int mis = 10000;
		int epoch = 0;
		while(mis > 350) {
			//if(true)break;
			System.out.println("==============================");
			System.out.println("TRAINING EPOCH #" + epoch + '\n');

			mis = nn.countMisclassifications(testFeatures, testLabels);
			System.out.println("Misclassifications: " + mis);

			for(int i = 0; i < trainFeatures.rows(); ++i) {
				Vec in, target;

				// Train the network on a single input
				in = trainFeatures.row(i);

				target = new Vec(10);
				target.vals[(int) trainLabels.row(i).get(0)] = 1;

				//nn.refineWeights(in, target, nn.weights, 0.0175, Training.STOCHASTIC);
			}

			// Shuffle training and testing indices
			for(int i = 0; i < trainingIndices.length * 0.5; ++i) {
				int randomIndex = random.nextInt(trainingIndices.length);
				int temp = trainingIndices[i];
				trainingIndices[i] = trainingIndices[randomIndex];
				trainingIndices[randomIndex] = temp;

			}

			for(int i = 0; i < testIndices.length * 0.5; ++i) {
				int randomIndex = random.nextInt(testIndices.length);
				int temp = testIndices[i];
				testIndices[i] = testIndices[randomIndex];
				testIndices[randomIndex] = temp;
			}

			++epoch;
		}
	}

	public static void testBackProp() {
		double[] x = {0, 1, 2};
		Vec xx = new Vec(x);
		double[] m = {1, 5, 1, 2, 3, 2, 1, 0};
		double[] yhat = {9, 6};
		LayerLinear ll = new LayerLinear(3, 2);
		ll.activate(new Vec(m), new Vec(x));
		ll.blame = new Vec(yhat);
		ll.backProp(new Vec(m), new Vec(x));
		System.out.println(xx);
	}

	public static void testGradient() {
		double[] x = {0, 1, 2};
		Vec xx = new Vec(x);
		double[] m = {1, 5, 1, 2, 3, 2, 1, 0};
		Vec mm = new Vec(m);
		Vec g = new Vec(mm.size());
		g.fill(0.0);
		double[] yhat = {9, 6};
		LayerLinear ll = new LayerLinear(3, 2);
		ll.activate(new Vec(m), new Vec(x));
		ll.blame = new Vec(yhat);
		ll.updateGradient(xx, g);
		//System.out.println(xx);
		System.out.println(g);
	}

	public static void testNomCat() {
		Random random = new Random(123456);

		/// Load data
		Matrix data = new Matrix();
		data.loadARFF("data/hypothyroid.arff");

		// Matrix data = new Matrix(5, derp.cols());
		// data.copyBlock(0, 0, derp, 0, 0, 5, data.cols());

		/// Create a new filter to preprocess our data
		Filter f = new Filter(random);

		/// Partition the features from the labels
		Matrix features = new Matrix();
		Matrix labels = new Matrix();
		f.splitLabels(data, features, labels);

		/// PREPROCESSING
		// We need a set of preprocessors for both features and labels

		// Train the preprocessors for the training data
		f.train(features, labels, null, 0, 0.0);

		/// Partition the data into training and testing blocks
		/// With respective feature and labels blocks
		double splitRatio = 0.75;
		Matrix trainingFeatures = new Matrix();
		Matrix trainingLabels = new Matrix();
		Matrix testingFeatures = new Matrix();
		Matrix testingLabels = new Matrix();
		f.splitData(features, labels, trainingFeatures, trainingLabels,
			testingFeatures, testingLabels, 5, 0);


		/// Build index arrays to shuffle training and testing data
		int[] trainingIndices = new int[trainingFeatures.rows()];
		int[] testIndices = new int[testingFeatures.rows()];

		// populate the index arrays with indices
		for(int i = 0; i < trainingIndices.length; ++i) { trainingIndices[i] = i; }
		for(int i = 0; i < testIndices.length; ++i) { testIndices[i] = i; }


		/// I want some intelligent way of getting the input and outputs
		f.nn.layers.add(new LayerLinear(trainingFeatures.cols(), 100));
		f.nn.layers.add(new LayerTanh(100));

		f.nn.layers.add(new LayerLinear(100, 4));
		f.nn.layers.add(new LayerTanh(4));

		f.nn.initWeights();

		int mis = testingLabels.rows();
		int epoch = 0;

		double testSSE = 0;
		double trainSSE = 0;

		double previous = 0;
		double tolerance = 0.0000009;


		System.out.println("batch,seconds,testRMSE,trainRMSE");
		int batch = 1;
		int batch_size = 10;
		double startTime = (double)System.nanoTime();
		while(true) {

			testSSE += f.sum_squared_error(testingFeatures, testingLabels);
			double testMSE = testSSE / testingFeatures.rows();
			double testRMSE = Math.sqrt(testMSE);

			trainSSE += f.sum_squared_error(trainingFeatures, trainingLabels);
			double trainMSE = trainSSE / trainingFeatures.rows();
			double trainRMSE = Math.sqrt(trainMSE);

			f.trainNeuralNet(trainingFeatures, trainingLabels, trainingIndices, batch_size, 0.0);
			// double mse = sse / batch;
			// double rmse = Math.sqrt(mse);

			double seconds = ((double)System.nanoTime() - startTime) / 1e9;
			System.out.println(batch + "," + seconds + "," + testRMSE + "," + trainRMSE);

			batch = batch + 1;

			// mis = f.countMisclassifications(testingFeatures, testingLabels);
			// System.out.println("mis: " + mis);

			double convergence = Math.abs(1 - (previous / testSSE));
			previous = testSSE;
			testSSE = 0;
			trainSSE = 0;
			if(convergence < tolerance) break;

		}

	}

	public static void debugSpew() {
		Random r = new Random(123456);
		NeuralNet nn = new NeuralNet(r);
		nn.layers.add(new LayerConv(new int[]{4, 4}, new int[]{3, 3, 2},
			new int[]{4, 4, 2}));
		nn.layers.add(new LayerLeakyRectifier(4 * 4 * 2));
		nn.layers.add(new LayerMaxPooling2D(4, 4, 2));

		double[] w = {
			0,							// bias #1
			0.1,						// bias #2

			0.01,0.02,0.03, // filter #1
			0.04,0.05,0.06,
			0.07,0.08,0.09,

			0.11,0.12,0.13, // filter #2
			0.14,0.15,0.16,
			0.17,0.18,0.19
		};
		nn.weights = new Vec(w);
		nn.gradient = new Vec(nn.weights.size());
		nn.gradient.fill(0.0);

		double[] in = {
			0,0.1,0.2,0.3,
			0.4,0.5,0.6,0.7,
			0.8,0.9,1,1.1,
			1.2,1.3,1.4,1.5
		};
		Vec input = new Vec(in);

		double[] t = {
			0.7,0.6,
			0.5,0.4,

			0.3,0.2,
			0.1,0
		};
		Vec target = new Vec(t);

		// Forward Prop
		nn.predict(input);
		System.out.println("activation 0:\n" + nn.layers.get(0).activation);
		nn.predict(input);
		System.out.println("activation 0:\n" + nn.layers.get(0).activation);
		// System.out.println("activation 1:\n" + nn.layers.get(1).activation);
		// System.out.println("activation 2:\n" + nn.layers.get(2).activation);
		//
		// // error
		// System.out.println("output Blame: ");
		// for(int i = 0; i < target.size(); ++i) {
		// 	System.out.print((target.get(i) - nn.layers.get(2).activation.get(i)) + ",");
		// }
		// System.out.println("");
		//
		// // backProp
		// nn.backProp(target);
		// System.out.println("blame 2: " + nn.layers.get(2).blame);
		// System.out.println("blame 1: " + nn.layers.get(1).blame);
		// System.out.println("blame 0: " + nn.layers.get(0).blame);
		//
		// nn.updateGradient(input);
		// System.out.println("gradient: " + nn.gradient);
		//
		// nn.cd_gradient = new Vec(nn.gradient.size());
		// nn.central_difference(input, target);
		// System.out.println("cd: " + nn.cd_gradient);
		//
		// int count = 0;
		// for(int i = 0; i < nn.gradient.size(); ++i) {
		// 	double difference = (nn.cd_gradient.get(i) - nn.gradient.get(i)) / nn.cd_gradient.get(i);
		// 	if(difference > 0.005)
		// 		++count;
		// }
		//
		// System.out.println("Difference exceeds tolerance " + count
		// 	+ " times out of " + nn.gradient.size() + " elements");

	}

	public static void debugSpew2() {
		Random r = new Random(123456);
		NeuralNet nn = new NeuralNet(r);

		nn.layers.add(new LayerConv(new int[]{4, 4}, new int[]{3, 3},
			new int[]{4, 4}));
		nn.layers.add(new LayerConv(new int[]{4, 4}, new int[]{3, 3, 2},
			new int[]{4, 4, 2}));
		nn.layers.add(new LayerLeakyRectifier(4 * 4 * 2));
		nn.layers.add(new LayerMaxPooling2D(4, 4, 2));

		double[] w = {
			0,							// bias #1

			0.01,0.02,0.03, // filter #1
			0.04,0.05,0.06,
			0.07,0.08,0.09,

			0.1,						// bias #2
			0.20,						// bias #3

			0.11,0.12,0.13, // filter #2
			0.14,0.15,0.16,
			0.17,0.18,0.19,

			0.21,0.22,0.23, // filter #3
			0.24,0.25,0.26,
			0.27,0.28,0.29
		};
		nn.weights = new Vec(w);
		nn.gradient = new Vec(nn.weights.size());
		nn.gradient.fill(0.0);

		double[] in = {
			0,0.1,0.2,0.3,
			0.4,0.5,0.6,0.7,
			0.8,0.9,1,1.1,
			1.2,1.3,1.4,1.5
		};
		Vec input = new Vec(in);

		double[] t = {
			0.7,0.6,
			0.5,0.4,

			0.3,0.2,
			0.1,0
		};
		Vec target = new Vec(t);

		nn.predict(input);
		System.out.println("activation 0:\n" + nn.layers.get(0).activation);
		System.out.println("activation 1:\n" + nn.layers.get(1).activation);
		System.out.println("activation 2:\n" + nn.layers.get(2).activation);
		System.out.println("activation 3:\n" + nn.layers.get(3).activation);

		// error
		System.out.println("output Blame: ");
		for(int i = 0; i < target.size(); ++i) {
			System.out.print((target.get(i) - nn.layers.get(3).activation.get(i)) + ",");
		}
		System.out.println("");

		// backProp
		nn.backProp(target);
		System.out.println("blame 2: " + nn.layers.get(2).blame);
		System.out.println("blame 1: " + nn.layers.get(1).blame);
		System.out.println("blame 0: " + nn.layers.get(0).blame);

		nn.updateGradient(input);
		System.out.println("gradient: " + nn.gradient);

		nn.refineWeights(0.01);
		System.out.println("weights: " + nn.weights);

		//Vec cd = nn.centralDifference(input);
		//System.out.println("cd: " + cd);
	}

	public static void asgn4() {
		/// Instantiate net
		Random r = new Random(123456);
		NeuralNet nn = new NeuralNet(r);

		/// Build topology
		nn.layers.add(new LayerConv(new int[]{8, 8}, new int[]{5, 5, 4}, new int[]{8, 8, 4}));
		nn.layers.add(new LayerLeakyRectifier(8 * 8 * 4));
		nn.layers.add(new LayerMaxPooling2D(8, 8, 4));
		nn.layers.add(new LayerConv(new int[]{4, 4, 4}, new int[]{3, 3, 4, 6}, new int[]{4, 4, 1, 6}));
		nn.layers.add(new LayerLeakyRectifier(4 * 4 * 6));
		nn.layers.add(new LayerMaxPooling2D(4, 4, 1 * 6));
		nn.layers.add(new LayerLinear(2 * 2 * 6, 3));
		nn.initWeights();

		/// Test data
		int inSize = nn.layers.get(0).inputs;
		Vec in = new Vec(inSize);
		for(int i = 0; i < in.size(); ++i) {
			in.set(i, i / 100.0);
		}

		int size = nn.layers.size();
		int outSize = nn.layers.get(size-1).outputs;
		Vec target = new Vec(outSize);
		for(int i = 0; i < target.size(); ++i) {
			target.set(i, i / 10.0);
		}

		nn.finite_difference(in, target);
	}

	public static void timeseries() {
		/// Instantiate net
		Random r = new Random(123456);
		NeuralNet nn = new NeuralNet(r);

		/// Build topology
		nn.layers.add(new LayerLinear(1, 101));
		nn.layers.add(new LayerSine(101));
		nn.layers.add(new LayerLinear(101, 1));
		nn.initWeights();

		/// Initilizize first layer weights
		int numWeights = nn.layers.get(0).getNumberWeights();
		// Strip the weights
		Vec layerOne = new Vec(nn.weights, 0, numWeights); // Strip the weights
		// Separate the bias and populate it
		Vec bias = new Vec(layerOne, 0, nn.layers.get(0).outputs);
		for(int i = 0; i < bias.size(); ++i) {
			if(i < 50)
				bias.set(i, Math.PI);
			else
				bias.set(i, Math.PI / 2);
		}

		// Separate M and populate bias
		Vec m = new Vec(layerOne, bias.size(), layerOne.size()-bias.size());
		for(int i = 0; i < m.size() - 1; ++i) {
			if(i < 50)
				m.set(i, (i+1) * 2 * Math.PI);
			else
				m.set(i, (i+1) * 2 * Math.PI);
		}
		m.set(m.size()-1, 0.01);

		/// Build the training features matrix
		Matrix trainingFeatures = new Matrix(256, 1);
		for(int i = 0; i < trainingFeatures.rows(); ++i) {
			trainingFeatures.row(i).set(0, i / 256.0);
		}

		/// Build the testing features matrix
		Matrix testingFeatures = new Matrix(100, 1);
		for(int i = 0; i < testingFeatures.rows(); ++i) {
			// testingFeatures.row(i).set(0, (256.0 + i) / 256.0);
			testingFeatures.row(i).set(0, (i) / 256.0);
		}

		/// Load label data from file
		Matrix data = new Matrix();
		data.loadARFF("data/unemployment.arff");

		/// split into training labels matrix
		Matrix trainingLabels = new Matrix(256, 1);
		for(int i = 0; i < trainingLabels.rows(); ++i) {
			double val = data.row(i).get(0);
			trainingLabels.row(i).set(0, val);
		}

		/// Split into testing labels matrix
		Matrix testingLabels = new Matrix (100, 1);
		for(int i = 0; i < testingLabels.rows(); ++i) {
			double val = data.row(256 + i).get(0);
			testingLabels.row(i).set(0, val);
		}

		/// Build index arrays to shuffle training and testing data
		int[] trainingIndices = new int[trainingFeatures.rows()];
		// populate the index arrays with indices
		for(int i = 0; i < trainingIndices.length; ++i) { trainingIndices[i] = i; }

		/// Train the net
		for(int i = 0; i < 1; ++i) {
			nn.train(trainingFeatures, trainingLabels, trainingIndices, 1, 0.0);
		}

		/// produce a matrix of the predicted results
		Vec predictions = new Vec(testingFeatures.rows());
		for(int i = 0; i < testingFeatures.rows(); ++i) {
			Vec pred = new Vec(nn.predict(testingFeatures.row(i)));
			predictions.set(i, pred.get(0));
		}

		for(int i = 0; i < predictions.size(); ++i) {
			System.out.println(predictions.get(i));
		}
	}


	public static void tsDebugSimple() {
		/// Instantiate net
		Random r = new Random(123456);
		NeuralNet nn = new NeuralNet(r);

		/// Build topology
		nn.layers.add(new LayerLinear(1, 5));
		nn.layers.add(new LayerSine(5));
		nn.layers.add(new LayerLinear(5, 1));

		double[] w = {
			3.1415926535898,3.1415926535898,1.5707963267949,1.5707963267949,
			0,6.2831853071796,12.566370614359,6.2831853071796,12.566370614359,0,
			0.01,0.01,0.01,0.01,0.01,0.01
		};

		// double[] w = {
		// 	3.1415926535898,3.1412587951824,1.5707963267949,1.5707963267949,0,6.2831853071796,
		// 	12.566370614359,6.2831853071796,12.566370614359,0,0.043385840734641,0.01,0.01,0.043385840734641,
		// 	0.043385840734641,0.11488471198587
		// };
		nn.weights = new Vec(w);
		nn.gradient = new Vec(nn.weights.size());

		double[] in = {0.0};
		Vec input = new Vec(in);

		double[] t = {3.4};
		Vec target = new Vec(t);

		nn.predict(input);
		nn.backProp(target);
		nn.updateGradient(input);

		System.out.println("activation 0:\n" + nn.layers.get(0).activation);
		System.out.println("activation 1:\n" + nn.layers.get(1).activation);
		System.out.println("activation 2:\n" + nn.layers.get(2).activation);

		System.out.println(nn.gradient);
	}

	public static void main(String[] args) {

		/// NOTE: l1 regularization pushes non-critical weights to 0.
		timeseries();
		//tsDebugSimple();
	}
}
