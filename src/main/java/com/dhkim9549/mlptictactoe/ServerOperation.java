package com.dhkim9549.mlptictactoe;

import jdk.nashorn.internal.parser.JSONParser;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.rmi.Naming;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import org.json.simple.*;
import org.json.simple.parser.*;

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
    public String helloTo(String boardStr) throws RemoteException {

        System.err.println("Client is trying to contact!");
        System.out.println("boardStr = " + boardStr);

        String jsonStr = boardStr;
        jsonStr = "{\"board\":" + jsonStr + "}";

        org.json.simple.parser.JSONParser jsonParser = new org.json.simple.parser.JSONParser();
        org.json.simple.JSONObject jsonObj = null;
        try {
            jsonObj = (org.json.simple.JSONObject) jsonParser.parse(jsonStr);
        } catch(Exception e) {
        }
        org.json.simple.JSONArray memberArray = (org.json.simple.JSONArray) jsonObj.get("board");

        System.out.println(jsonStr);
        System.out.println(memberArray.size());
        for(int i=0 ; i<memberArray.size() ; i++){
            System.out.print(memberArray.get(i) + ",");
        }


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
            System.setProperty("java.rmi.server.hostname","bada.ai");
            Naming.rebind("//localhost/MyServer", new ServerOperation());
            System.err.println("Server ready");

        } catch (Exception e) {

            System.err.println("Server exception: " + e.toString());
            e.printStackTrace();

        }

    }

}
