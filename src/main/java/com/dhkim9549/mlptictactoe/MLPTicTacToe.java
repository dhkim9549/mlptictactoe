package com.dhkim9549.mlptictactoe;

import java.io.*;
import java.util.*;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 *  Tic Tac Toe Reinforcement Learning Example
 *
 * @author Dong-Hyun Kim
 */
public class MLPTicTacToe {

    public static void main(String[] args) throws Exception {

        long seed = 123;
        int batchSize = 50;

        MultiLayerNetwork model = getInitModel();
        //model = readModelFromFile("/down/ttt_model.zip");

        //Load the training data:
        List<DataSet> listDs = getTrainingData(new Random(), model);
        DataSetIterator trainIter = new ListDataSetIterator(listDs, batchSize);

        model = train(model, trainIter);
        System.out.println("model = " + model);

        writeModelToFile(model, "/down/ttt_model.zip");

        /*
        for(int i = 0; i < 100; i++) {
            double[][] data = new double[1][9];
            data[0][0] = (double)i / 100.0;
            data[0][1] = 0.777;
            INDArray features = new NDArray(data);
            System.out.println();
            System.out.println("features = " + features);
            INDArray predicted = model.output(features, false);
            System.out.println("predicted = " + predicted);
        }
        */
    }

    public static MultiLayerNetwork getInitModel() throws Exception {

        int seed = 123;
        double learningRate = 0.003;

        int numInputs = 9 * 3;
        int numOutputs = 9;
        int numHiddenNodes = 9;

        //log.info("Build model....");
        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        return model;
    }

    public static MultiLayerNetwork train(MultiLayerNetwork model, DataSetIterator trainIter) throws Exception {

        model.setListeners(new ScoreIterationListener(100));    //Print score every 100 parameter updates

        model.fit( trainIter );

        return model;
    }

    private static List<DataSet> playGame(MultiLayerNetwork model) {

        GameState gs = new GameState();

        return null;
    }

    private static List<DataSet> getTrainingData(Random rand, MultiLayerNetwork model) {

        int nSamples = 10;
        int numInputs = 9 * 3;
        int numOutputs = 9;

        List<DataSet> listDs = new ArrayList<DataSet>();

        for (int i = 0; i < nSamples; i++) {

            List<DataSet> ds = playGame(model);
            listDs.addAll(ds);
        }

        System.out.println("listDs.size() = " + listDs.size());
        Collections.shuffle(listDs, rand);

        System.out.println("listDs = " + listDs);

        return listDs;
    }

    public static MultiLayerNetwork readModelFromFile(String fileName) throws Exception {

        System.out.println("Deserializing model...");

        // Load the model
        File locationToSave = new File(fileName);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        System.out.println("Deserializing model complete.");

        return model;
    }

    public static void writeModelToFile(MultiLayerNetwork model, String fileName) throws Exception {

        System.out.println("Serializing model...");

        // Save the model
        File locationToSave = new File(fileName); // Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true; // Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);

        System.out.println("Serializing model complete.");

    }

}
