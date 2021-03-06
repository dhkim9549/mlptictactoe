package com.dhkim9549.mlptictactoe;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.json.simple.*;
import org.json.simple.parser.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.rmi.Naming;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.util.Date;

/**
 * Created by user on 2017-07-02.
 */
public class ServerOperation extends UnicastRemoteObject implements RMIInterface {

    private static final long serialVersionUID = 1L;

    MultiLayerNetwork model = null;

    protected ServerOperation() throws RemoteException {

        super();
        try {
            //model = MLPTicTacToe.readModelFromFile("/down/sin/ttt_model_h2_uSGD_mb16_ss16_5210000.zip");
            model = MLPTicTacToe.readModelFromFile("/down/sin/ttt_model_h2_uSGD_ge_mb16_ss16_ev100000_aRP_5780000.zip");
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    @Override
    public String helloTo(String boardJsonStr) throws RemoteException {

        System.out.println("\nClient is trying to contact! ver 2.0");
        System.out.println(new Date());
        System.out.println("boardJsonStr = " + boardJsonStr);

        JSONParser jsonParser = new JSONParser();
        JSONObject jsonObj = null;
        try {
            jsonObj = (JSONObject) jsonParser.parse(boardJsonStr);
            System.out.println("jsonObj = " + jsonObj);
        } catch(Exception e) {
            System.out.println(e);
        }
        JSONArray memberArray = (JSONArray)jsonObj.get("board");
        String currentPlayerStr = (String)jsonObj.get("currentPlayer");
        System.out.println("memberArray.size() = " + memberArray.size());

        GameState gs = new GameState();

        int currentPlayer = 0;
        if(currentPlayerStr.equals("X")) {
            currentPlayer = 1;
        } else if(currentPlayerStr.equals("O")) {
            currentPlayer = -1;
        }
        gs.currentPlayer = currentPlayer;

        for(int i = 0 ; i < memberArray.size() ; i++) {
            int b = 0;
            String m = (String)memberArray.get(i);
            if(m.equals("X")) {
                b = 1;
            } else if(m.equals("O")) {
                b = -1;
            }
            gs.board[i / 3][i % 3] = b;
        }

        System.out.println("gs = " + gs);

        INDArray output = model.output(gs.getFeature(gs.getCurrentPlayer()));
        System.out.println("output = " + output);

        Policy policy = new Policy(model, 0.0);
        int a = policy.chooseMove(gs, true);
        System.out.println("a = " + a);

        JSONObject moveJsonObj = new JSONObject();
        moveJsonObj.put("a", a);
        System.out.println("moveJsonObj.toString() = " + moveJsonObj.toString());

        return moveJsonObj.toString();
    }

    public static void main(String[] args) {

        try {

            System.err.println("Starting.....");
            System.setProperty("java.rmi.server.hostname","bada.ai");
            Naming.rebind("MyServer", new ServerOperation());
            System.err.println("Server ready");

        } catch (Exception e) {

            System.err.println("Server exception: " + e.toString());
            e.printStackTrace();

        }

    }

}
