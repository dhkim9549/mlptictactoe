package com.dhkim9549.mlptictactoe;

import java.net.MalformedURLException;
import java.rmi.Naming;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.io.*;

/**
 * Created by user on 2017-07-02.
 */
public class ClientOperation {

    private static RMIInterface look_up;

    public static void main(String[] args)
            throws MalformedURLException, RemoteException, NotBoundException {

        look_up = (RMIInterface) Naming.lookup("//bada.ai/MyServer");

        String message = "";

        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        try {
            System.out.print("Enter message:");
            message = br.readLine();
        } catch (Exception e){
            e.printStackTrace();
        }

        String response = look_up.helloTo(message);
        System.out.print(response);

    }

}