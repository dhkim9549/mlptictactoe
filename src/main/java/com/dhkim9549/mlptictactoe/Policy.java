package com.dhkim9549.mlptictactoe;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

/**
 * Behavior policy
 */
public class Policy {

    private double epsilon = 0.0;
    private MultiLayerNetwork model = null;
    private boolean toBeTrained = false;

    public Policy(MultiLayerNetwork model, double epsilon) {

        this.model = model;
        this.epsilon = epsilon;
    }

    public Policy(MultiLayerNetwork model, double epsilon, boolean toBeTrained) {

        this.model = model;
        this.epsilon = epsilon;
        this.toBeTrained = toBeTrained;
    }

    /**
     * Chooses the best move for the next player given a game state
     * @param printOption
     * @return the best move
     */
    public int chooseMove(GameState gs, boolean printOption) {

/*
        if(nextPlayer == -1) {
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            try {
                System.out.println("gs = " + this);
                System.out.print("Enter Integer:");
                int aa = Integer.parseInt(br.readLine());
                System.out.println("gs = " + this);
                return aa;
            } catch (Exception e){
                e.printStackTrace();
            }
        }
*/

        int a = -1;
        Random rnd = new Random();

        if(rnd.nextDouble() < epsilon) {
            while (a == -1) {
                int i = rnd.nextInt(9);
                if (gs.board[i / 3][i % 3] == 0) {
                    a = i;
                }
            }

            return a;
        }

        double max = - 1.0;

        INDArray valueArray = Nd4j.create(3, 3);

        for(int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (gs.board[i][j] != 0) {
                    continue;
                }
                gs.board[i][j] = gs.nextPlayer;
                INDArray feature = gs.getFeature(gs.nextPlayer);
                INDArray outputArray = model.output(feature);
                double v = outputArray.getDouble(0, 0);
                valueArray.put(i, j, v);
                if (max < v) {
                    a = i * 3 + j;
                    max = v;
                }
                gs.board[i][j] = 0;
            }
        }

        if(printOption) {
            System.out.println("valueArray = " + valueArray);
        }

        return a;
    }

    /**
     *
     * @return Returns whether this policy is to be trained or not
     */
    public boolean isToBeTrained() {
        return toBeTrained;
    }
}
