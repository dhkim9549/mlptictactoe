package com.dhkim9549.mlptictactoe;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.util.*;

/**
 *  Tic Tac Toe Reinforcement Learning
 * @author Dong-Hyun Kim
 */
public class MLPTicTacToe {

    private static final Logger log = LoggerFactory.getLogger(MLPTicTacToe.class);

    static String hpId = "h2_uSGD_ge_mb16_ss16_ev100000_aSP";

    //double learnigRate = Double.parseDouble(args[0]);
    static double learnigRate = 0.0025;

    // Number of sample size per iteration
    static long nSamples = 16;

    // Mini-batch size
    static int batchSize = 16;

    // Evaluation sample size
    static long nEvalSamples = 10000;

    public static void main(String[] args) throws Exception {

        log.info("************************************************");
        log.info("hpId = " + hpId);
        log.info("Number of hidden layers = 2");
        log.info("learnigRate = " + learnigRate);
        log.info("Updater = " + "SGD");
        log.info("mini-batch size (batchSize) = " + batchSize);
        log.info("Number of sample size per iteration (nSamples) = " + nSamples);
        log.info("i >= 0");
        log.info("************************************************");

        MultiLayerNetwork model = getInitModel(learnigRate);
        //MultiLayerNetwork model = readModelFromFile("/down/ttt_model_h2_uSGD_ge_mb16_ss16_ev100000_aRP_3200000.zip");
        //MultiLayerNetwork model = readModelFromFile("/down/ttt_model_h2_uSGD_mb16_ss16_5210000.zip");

        ArrayList<Policy> opponentPool = new ArrayList<>();

        NeuralNetConfiguration config = model.conf();
        log.info("config = " + config);

        // training iteration
        long i = 0;

        long lastIterationModelSave = 0;

        // greedy epsilon
        double epsilon = 0.0;

        while(true) {

            if(i % 5000 == 0 || lastLosingRate == 0.0) {
                log.info("Date = " + new Date());
                evaluate(new Policy(model, 0.0, true), new SupervisedPolicy());
                evaluateModel(model);
            }

            if(opponentPool.isEmpty()) {
                //opponentPool.add(new RandomPolicy());
                opponentPool.add(new SupervisedPolicy());
            }

            // If the model never loses during the evaluation, the training stops.
            if(lastLosingRate != 0.0) {

                i++;
                epsilon = Math.max(0.1, 1.0 - (double)i / 100000.0);

                if(i % 1000 == 0) {
                    log.info("Training count i = " + i);
                    log.info("epsilon = " + epsilon);
                }

                // Load the training data.
                List<DataSet> listDs = getTrainingData(new Policy(model, epsilon, true), opponentPool);
                DataSetIterator trainIter = new ListDataSetIterator(listDs, batchSize);

                // Train the model
                model = train(model, trainIter);
            }

            if(lastIterationModelSave != i) {
                if (i % 10000 == 0 || lastLosingRate == 0.0) {
                    writeModelToFile(model, "/down/ttt_model_" + hpId + "_" + i + ".zip");
                    lastIterationModelSave = i;
                }
            }
        }
    }

    public static MultiLayerNetwork getInitModel(double learningRate) throws Exception {

        int seed = 123;

        int numInputs = 9 * 3;
        int numOutputs = 2;
        int numHiddenNodes = 9 * 3;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.SGD)
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

        //model.setListeners(new ScoreIterationListener(500));

        model.fit( trainIter );

        return model;
    }

    private static List<DataSet> playGame(Policy policy1, Policy policy2) {

        List<DataSet> dsList = new ArrayList<>();
        List<INDArray> featureArray = new ArrayList<>();
        List<Integer>moveList = new ArrayList<>();

        GameState gs = new GameState();

        while(!gs.isOver()) {

            int a = 0;
            if(gs.getCurrentPlayer() == 1) {
                a = policy1.chooseMove(gs, false);
            } else {
                a = policy2.chooseMove(gs, false);
            }
            moveList.add(a);

            gs.playMove(a);

            featureArray.add(gs.getFeature(- gs.getCurrentPlayer()));
        }

/*
        if(gs.getWinner() == -1 && policy1.isToBeTrained()) {
            log.info("LOSE **************");
            log.info("moveList = " + moveList);
        }
        if(gs.getWinner() == 1 && policy2.isToBeTrained()) {
            log.info("LOSE **************");
            log.info("moveList = " + moveList);
        }
*/
/*

        log.info("gs = " + gs);
        log.info("gggggggggggggggggggggggg\n\n");
*/

        int winner = gs.getWinner();
        int player = 1;

        for (INDArray feature: featureArray) {
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
            DataSet ds = new DataSet(feature, label);
            if(player == 1 && policy1.isToBeTrained()) {
                dsList.add(ds);
            }
            if(player == -1 && policy2.isToBeTrained()) {
                dsList.add(ds);
            }
            player *= -1;
        }

        return dsList;
    }

    private static List<DataSet> getTrainingData(Policy playerPolicy, ArrayList<Policy> opponentPool) {

        //log.info("Getting training data...");
        //log.info("opponentPool.size() = " + opponentPool.size());

        Random rnd = new Random();

        List<DataSet> listDs = new ArrayList<>();

        for (int i = 0; i < nSamples; i++) {

            // Pick a opponent randomly from the opponent pool.
            Policy opponentPolicy = opponentPool.get(rnd.nextInt(opponentPool.size()));

            List<DataSet> ds = null;

            if(i % 2 == 0) {
                ds = playGame(playerPolicy, opponentPolicy);
            } else {
                ds = playGame(opponentPolicy, playerPolicy);
            }

            // Randomly choose one data set from 'ds' and discard the rest.
            listDs.add(ds.get(rnd.nextInt(ds.size())));
        }

        Collections.shuffle(listDs);

        //log.info("listDs.size() = " + listDs.size());
        //log.info("Getting training complete.");

        return listDs;
    }

    public static MultiLayerNetwork readModelFromFile(String fileName) throws Exception {

        log.info("Deserializing model...");

        // Load the model
        File locationToSave = new File(fileName);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        log.info("Deserializing model complete.");

        return model;
    }

    public static void writeModelToFile(MultiLayerNetwork model, String fileName) throws Exception {

        log.info("Serializing model...");

        // Save the model
        File locationToSave = new File(fileName); // Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true; // Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);

        log.info("Serializing model complete.");

    }

    public static void evaluateModel(MultiLayerNetwork model) {

        // Evaluate
        {
            GameState gs = new GameState();

            INDArray output = model.output(gs.getFeature(gs.getCurrentPlayer()));
            log.info("output = " + output);

            Policy policy = new Policy(model, 0.0);
            int a = policy.chooseMove(gs, true);
            log.info("a = " + a);
        }

        // Evaluate 2
        {
            GameState gs = new GameState();
            gs.playMove(1);
            gs.playMove(2);
            gs.playMove(4);
            gs.playMove(5);

            INDArray output = model.output(gs.getFeature(gs.getCurrentPlayer()));
            log.info("output = " + output);

            Policy policy = new Policy(model, 0.0);
            int a = policy.chooseMove(gs, true);
            log.info("a = " + a);
        }

        // Evaluate 3
        {
            GameState gs = new GameState();
            gs.playMove(0);
            gs.playMove(4);
            gs.playMove(1);

            INDArray output = model.output(gs.getFeature(gs.getCurrentPlayer()));
            log.info("output = " + output);

            Policy policy = new Policy(model, 0.0);
            int a = policy.chooseMove(gs, true);
            log.info("a = " + a);
        }

        // Evaluate 4
        {
            GameState gs = new GameState();
            gs.playMove(2);
            gs.playMove(4);
            gs.playMove(3);

            INDArray output = model.output(gs.getFeature(gs.getCurrentPlayer()));
            log.info("output = " + output);

            Policy policy = new Policy(model, 0.0);
            int a = policy.chooseMove(gs, true);
            log.info("a = " + a);
        }
    }

    public static ArrayList<Policy> loadOpponentPoolFromFiles() throws Exception {

        ArrayList<Policy> pool = new ArrayList<>();

        for(int i = 100; i <= 300; i += 10) {
            String fileName = "/down/ttt_model_" + i + ".zip";
            MultiLayerNetwork model = readModelFromFile(fileName);
            Policy p = new Policy(model, 0.1);
            pool.add(p);
        }

        return pool;
    }

    static double lastLosingRate = - 1.0;

    private static void evaluate(Policy playerPolicy, Policy opponentPolicy) {

        log.info("Evaluating...");

        int numOfWins = 0;
        int numOfLosses = 0;
        int numOfPlays = 0;

        for (int i = 0; i < nEvalSamples; i++) {

            List<DataSet> ds = null;

            if(i % 2 == 0) {
                ds = playGame(playerPolicy, opponentPolicy);
            } else {
                ds = playGame(opponentPolicy, playerPolicy);
            }

            Double label = ds.get(0).getLabels().getDouble(0);
            if(label == 1.0) {
                numOfWins++;
            } else if(label == 0.0) {
                numOfLosses++;
            }
            numOfPlays++;
        }

        lastLosingRate = (double)numOfLosses / (double)numOfPlays;

        log.info("\n");
        log.info("*** Evaluation Result ***");
        log.info(hpId + ": ");
        log.info("Winning rate = " + (double)numOfWins / (double)numOfPlays + ", ");
        log.info("Losing rate = " + lastLosingRate  + "\n");
        log.info("*************************");
        log.info("\n");
    }
}
