package com.dhkim9549.mlptictactoe;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * Created by user on 2017-07-01.
 */
public class PlayTicTacToe {

    public static void main(String[] args) throws Exception {

        MultiLayerNetwork model = MLPTicTacToe.readModelFromFile("/down/ttt_model_h2_uSGD_mb16_ss16_5210000.zip");

        int i = 0;

        while (true) {

            Policy aiPolicy = new Policy(model, 0.0);

            if(i % 2 == 0) {
                playGame(new HumanPolicy(), aiPolicy);
            } else {
                playGame(aiPolicy, new HumanPolicy());
            }
            i++;
        }
    }


    private static void playGame(Policy policy1, Policy policy2) {

        GameState gs = new GameState();

        while(!gs.isOver()) {

            int a = 0;
            if(gs.getCurrentPlayer() == 1) {
                a = policy1.chooseMove(gs, false);
            } else {
                a = policy2.chooseMove(gs, false);
            }

            gs.playMove(a);
        }

        System.out.println("gs = " + gs);
        System.out.println("*************************");
        System.out.println("Game Over");
        System.out.println("*************************");
    }
}
