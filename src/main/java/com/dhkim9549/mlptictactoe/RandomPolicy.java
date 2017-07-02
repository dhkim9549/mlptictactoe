package com.dhkim9549.mlptictactoe;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Random;

/**
 * Created by user on 2017-06-24.
 */
public class RandomPolicy extends Policy {

    RandomPolicy() {
        super(null, 0.0);
    }

    /**
     * Choose a random move
     * @param printOption
     * @return the best move
     */
    public int chooseMove(GameState gs, boolean printOption) {

        int a = -1;

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
