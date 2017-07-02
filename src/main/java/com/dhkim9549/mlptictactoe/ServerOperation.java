package com.dhkim9549.mlptictactoe;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.rmi.Naming;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

/**
 * Created by user on 2017-07-02.
 */
public class ServerOperation extends UnicastRemoteObject implements RMIInterface {

    private static final long serialVersionUID = 1L;

    MultiLayerNetwork model = null;

    protected ServerOperation() throws RemoteException {

        super();
        try {
            model = MLPTicTacToe.readModelFromFile("/down/ttt_model_h2_uSGD_mb16_ss16_5210000.zip");
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    @Override
    public String helloTo(String name) throws RemoteException {

        System.err.println(name + " is trying to contact!");

        GameState gs = new GameState();
        gs.playMove(2);
        gs.playMove(4);
        gs.playMove(3);


        INDArray output = model.output(gs.getFeature(gs.getNextPlayer()));
        System.out.println("output = " + output);

        Policy policy = new Policy(model, 0.0);
        int a = policy.chooseMove(gs, true);
        System.out.println("a = " + a);


        return "a = " + a;

    }

    public static void main(String[] args) {

        try {

            System.err.println("Starting.....");
            System.setProperty("java.rmi.server.hostname","220.230.113.107");
            Naming.rebind("//localhost/MyServer", new ServerOperation());
            System.err.println("Server ready");

        } catch (Exception e) {

            System.err.println("Server exception: " + e.toString());
            e.printStackTrace();

        }

    }

}
