package com.dhkim9549.mlptictactoe;

import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 * Created by user on 2017-06-22.
 */
public class HumanPolicy extends Policy {

    HumanPolicy() {
        super(null, 0.0);
    }

    /**
     * Chooses a move from the keyboard
     * @param printOption
     * @return the best move
     */
    public int chooseMove(GameState gs, boolean printOption) {

        int a = -1;

        System.out.println("gs = " + gs);
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        try {
            System.out.print("Enter Integer:");
            a = Integer.parseInt(br.readLine());
        } catch (Exception e){
            e.printStackTrace();
        }

        return a;
    }
}
