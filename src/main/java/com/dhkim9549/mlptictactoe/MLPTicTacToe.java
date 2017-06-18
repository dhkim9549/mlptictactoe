package com.dhkim9549.mlptictactoe;

import java.io.*;
import java.util.*;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 *  Tic Tac Toe Reinforcement Learning Example
 *
 * @author Dong-Hyun Kim
 */
public class MLPTicTacToe {

    public static void main(String[] args) throws Exception {

        int batchSize = 15;

        //MultiLayerNetwork model = getInitModel();
        MultiLayerNetwork model = readModelFromFile("/down/ttt_model_310_2.zip");
        //MultiLayerNetwork oponentModel = readModelFromFile("/down/ttt_model_200_2.zip");

        NeuralNetConfiguration config = model.conf();
        System.out.println("config = " + config);

        for(int i = 311; i < 10000; i++) {

            // Evaluate
            {
                GameState gs = new GameState();

                INDArray output = model.output(gs.getFeature(gs.getNextPlayer()));
                System.out.println("output = " + output);

                int a = gs.chooseMove(model, true, false);
                System.out.println("a = " + a);
            }

            // Evaluate 2
            {
                GameState gs = new GameState();
                gs.playMove(1);
                gs.playMove(2);
                gs.playMove(4);
                gs.playMove(5);

                INDArray output = model.output(gs.getFeature(gs.getNextPlayer()));
                System.out.println("output = " + output);

                int a = gs.chooseMove(model, true, false);
                System.out.println("a = " + a);
            }

            System.out.println("Training count i = " + i);

            //Load the training data:
            List<DataSet> listDs = getTrainingData(model, model);

            DataSetIterator trainIter = new ListDataSetIterator(listDs, batchSize);
            //model = train(model, trainIter);
            System.out.println("\n\n");
            System.out.println("model = " + model);

            if(i % 10 == 0) {
                writeModelToFile(model, "/down/ttt_model_" + i + ".zip");
            }
        }
    }

    public static MultiLayerNetwork getInitModel() throws Exception {

        int seed = 123;
        double learningRate = 0.003;

        int numInputs = 9 * 3;
        int numOutputs = 2;
        int numHiddenNodes = 9 * 3;

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
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

    private static int numOfWins = 0;
    private static int numOfLoses = 0;
    private static int numOfPlays = 0;

    private static List<DataSet> playGame(MultiLayerNetwork model, MultiLayerNetwork opponentModel) {

        List<DataSet> dsList = new ArrayList<>();
        List<INDArray> featureArray = new ArrayList<>();
        List moveList = new ArrayList();

        GameState gs = new GameState();

        while(!gs.isOver()) {

            int a = 0;
            if(gs.getNextPlayer() == 1) {
                a = gs.chooseMove(model, false, true);
            } else {
                a = gs.chooseMove(opponentModel, false, true);
            }
            moveList.add(a);

            gs.playMove(a);

            featureArray.add(gs.getFeature(- gs.getNextPlayer()));
        }

        if(gs.getWinner() == -1) {
            System.out.println("LOSE **************");
            System.out.println("moveList = " + moveList);
        }


/*
        System.out.println("gs = " + gs);
        System.out.println("gggggggggggggggggggggggg\n\n");
*/

        numOfPlays++;
        if(gs.getWinner() == 1) {
            numOfWins++;
        } else if(gs.getWinner() == -1) {
            numOfLoses++;
        }

        int winner = gs.getWinner();
        int player = 1;
        if(true || winner != 0) {
            Iterator it = featureArray.iterator();
            while (it.hasNext()) {
                double[] labelData = new double[2];
                if (winner == player) {
                    labelData[0] = 1.0;
                    labelData[1] = 0.0;
                } else if (winner == - player) {
                    labelData[0] = 0.0;
                    labelData[1] = 1.0;
                } else {
                    labelData[0] = 0.5;
                    labelData[1] = 0.5;
                }
                INDArray label = Nd4j.create(labelData, new int[]{1, 2});
                DataSet ds = new DataSet((INDArray) it.next(), label);
                if(player == 1) {
                    dsList.add(ds);
                }
                player *= -1;
            }
        }

        return dsList;
    }

    private static List<DataSet> getTrainingData(MultiLayerNetwork model, MultiLayerNetwork opponentModel) {

        int nSamples = 5000;

        List<DataSet> listDs = new ArrayList<DataSet>();

        for (int i = 0; i < nSamples; i++) {

            List<DataSet> ds = playGame(model, opponentModel);
            listDs.addAll(ds);
        }

        System.out.println("listDs.size() = " + listDs.size());
        Collections.shuffle(listDs, new Random());

        List<DataSet> listDs2 = new ArrayList<DataSet>();
        Iterator it = listDs.iterator();
        int cnt = 0;
        while(it.hasNext() && cnt < listDs.size() * 0.15) {
            listDs2.add((DataSet)it.next());
            cnt++;
        }
        System.out.println("listDs2.size() = " + listDs2.size());
        System.out.println("Winning rate = " + (double)numOfWins / (double)numOfPlays);
        System.out.println("Losing rate = " + (double)numOfLoses / (double)numOfPlays);

        return listDs2;
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
