package com.dhkim9549.mlptictactoe;

import java.util.Random;
import java.io.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
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
    public int[][] board;

    /**
     * Winner if the game is at the terminal state.
     * Either 1 or -1
     */
    private int winner;

    /**
     * Next player
     * Either 1 or -1
     */
    public int nextPlayer;

    public GameState() {
        board = new int[3][3];
        winner = 0;
        nextPlayer = 1;
    }

    public static void main(String[] args) {

        GameState gs = new GameState();
        gs.placeStone(0, 2);
        gs.placeStone(0, 0);
        gs.placeStone(1, 2);
        gs.placeStone(1, 1);
        System.out.println("gs = \n" + gs);
        gs.placeStone(2, 2);
        System.out.println("gs = \n" + gs);
    }

    public String toString() {

        double[][] boardDouble = new double[3][3];
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                boardDouble[i][j] = (double)board[i][j];
            }
        }

        INDArray bArray = Nd4j.create(boardDouble);

        String s = "";
        s += "\n" + bArray.toString();
        s += "\nisOver = " + isOver();
        s += "\nnextPlayer = " + nextPlayer;
        s += "\nwinner = " + winner;

        return s;
    }

    /**
     * Get a feature for the neural network input.
     * @param player
     * @return
     */
    public INDArray getFeature(int player) {

        double[] playerData = new double[9];
        double[] opponentData = new double[9];
        double[] emptyData = new double[9];

        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                if(board[i][j] == player) {
                    playerData[i * 3 + j] = 1.0;
                } else if(board[i][j] == - player) {
                    opponentData[i * 3 + j] = 1.0;
                } else if(board[i][j] == 0) {
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
     * Play the next move
     * @param a action
     */
    public void playMove(int a) {

        placeStone(a / 3, a % 3);
    }

    /**
     * Check if the game is over.
     */
    public boolean isOver() {

        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {

                int basis = board[i][j];
                if(basis == 0) {
                    continue;
                }

                // Check diagonally 1
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

                // Check diagonally 2
                for(int k = 0; k < 3; k++) {
                    int m = i + k;
                    int n = j - k;
                    if(m >= 3 || n >= 3) {
                        break;
                    }
                    if(m < 0 || n < 0) {
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
        int cnt = 0;
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                if(board[i][j] != 0) {
                    cnt++;
                }
            }
        }
        if(cnt == 9) {
            return true;
        }

        return false;
    }

    /**
     * Returns the winner of the game
     * @return
     */
    public int getWinner() {

        return winner;
    }

    /**
     * Returns the next player of the game
     * @return
     */
    public int getNextPlayer() {

        return nextPlayer;
    }
}
