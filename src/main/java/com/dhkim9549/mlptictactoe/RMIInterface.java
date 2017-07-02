package com.dhkim9549.mlptictactoe;

import java.rmi.Remote;
import java.rmi.RemoteException;

/**
 * Created by user on 2017-07-02.
 */
public interface RMIInterface extends Remote {

    public String helloTo(String name) throws RemoteException;

}