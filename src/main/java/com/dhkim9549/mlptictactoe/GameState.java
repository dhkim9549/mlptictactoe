package com.dhkim9549.mlptictactoe;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Represents a game state of Tic Tac Toe.
 */
public class GameState {

    /**
     * Board state
     * Either 1 or -1
     * 0 if empty
     */
    private int[][] board;

    /**
     * Winner if the game is at the terminal state.
     * Either 1 or -1
     */
    private int winner;

    /**
     * Next player
     * Either 1 or -1
     */
    private int nextPlayer;

    public GameState() {
        board = new int[3][3];
        winner = 0;
        nextPlayer = 1;
    }

    public static void main(String[] args) {

        GameState gs = new GameState();
        gs.placeStone(0, 0);
        gs.placeStone(0, 1);
        System.out.println("gs = \n" + gs);
        System.out.println("features = " + gs.getFeature());

    }

    public String toString() {

        double[][] boardDouble = new double[3][3];
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                boardDouble[i][j] = (double)board[i][j];
            }
        }

        INDArray bArray = Nd4j.create(boardDouble);

        String s = bArray.toString();
        s += "\nisOver = " + isOver();
        s += "\nnextPlayer = " + nextPlayer;
        s += "\nwinner = " + winner;

        return s;
    }

    /**
     * Get a feature for the neural network input.
     */
    public INDArray getFeature() {

        double[] playerData = new double[9];
        double[] opponentData = new double[9];
        double[] emptyData = new double[9];

        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                if(board[i][j] == this.nextPlayer) {
                    playerData[i * 3 + j] = 1.0;
                } else if(board[i][j] == - this.nextPlayer) {
                    opponentData[i * 3 + j] = 1.0;
                } else {
                    emptyData[i * 3 + j] = 1.0;
                }
            }
        }

        INDArray playerArray = Nd4j.create(playerData, new int[]{1, 9});
        INDArray opponentArray = Nd4j.create(opponentData, new int[]{1, 9});
        INDArray emptyArray = Nd4j.create(emptyData, new int[]{1, 9});

        INDArray featureArray = Nd4j.hstack(playerArray, opponentArray, emptyArray);

        return featureArray;
    }

    /**
     * Place a stone on the board of the next player.
     */
    public void placeStone(int x, int y) {

        this.board[x][y] = nextPlayer;
        nextPlayer = - nextPlayer;
    }

    /**
     * Check if the game is over.
     */
    public boolean isOver() {

        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                int basis = board[i][j];
                if(basis == 0) {
                    break;
                }

                // Check diagonally
                for(int k = 0; k < 3; k++) {
                    int m = i + k;
                    int n = j + k;
                    if(m >= 3 || n >= 3) {
                        break;
                    }
                    if(basis != board[m][n]) {
                        break;
                    }
                    if(k == 2) {
                        winner = basis;
                        return true;
                    }
                }

                // Check horizontally
                for(int k = 0; k < 3; k++) {
                    int m = i + k;
                    int n = j;
                    if(m >= 3 || n >= 3) {
                        break;
                    }
                    if(basis != board[m][n]) {
                        break;
                    }
                    if(k == 2) {
                        winner = basis;
                        return true;
                    }
                }

                // Check vertically
                for(int k = 0; k < 3; k++) {
                    int m = i;
                    int n = j + k;
                    if(m >= 3 || n >= 3) {
                        break;
                    }
                    if(basis != board[m][n]) {
                        break;
                    }
                    if(k == 2) {
                        winner = basis;
                        return true;
                    }
                }
            }
        }

        // Check if the board is full
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                if(board[i][j] == 0) {
                    break;
                }
                if(i == 2 && j == 2) {
                    return true;
                }
            }
        }

        return false;
    }
}
