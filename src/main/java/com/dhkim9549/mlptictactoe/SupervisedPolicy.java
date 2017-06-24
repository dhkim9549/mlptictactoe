package com.dhkim9549.mlptictactoe;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Random;

/**
 * Created by user on 2017-06-24.
 */
public class SupervisedPolicy extends Policy {

    SupervisedPolicy() {
        super(null, 0.0);
    }

    /**
     * Choose a move which wins the game with one move if possible, otherwise choose a random move
     * @param printOption
     * @return the best move
     */
    public int chooseMove(GameState gs, boolean printOption) {

        int a = -1;

        for(int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (gs.board[i][j] != 0) {
                    continue;
                }
                gs.board[i][j] = gs.nextPlayer;
                if (gs.isOver() && gs.getWinner() == gs.nextPlayer) {
                    a = i * 3 + j;
                }
                gs.board[i][j] = 0;
            }
        }

        if(a >= 0) {
            return a;
        }

        Random rnd = new Random();

        while (a == -1) {
            int i = rnd.nextInt(9);
            if (gs.board[i / 3][i % 3] == 0) {
                a = i;
            }
        }

        return a;
    }
}
