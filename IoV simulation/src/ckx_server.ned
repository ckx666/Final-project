//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

package ckx;

import inet.applications.contract.IApp;

simple Ckx_server like IApp {
    parameters:
        int localPort = default(-1);
        int destPort = default(-1);
     	int server_eta_threshold;
        string destAddress = default("car[" + string(ancestorIndex(0)) + "]");
        string interfaceTableModule;
        string multicastInterface = default("wlan");
        
        double period @unit("s")= default(0.01s);
    gates:
        output socketOut;
        input socketIn;
        input hostIn @directIn;
        output hostOut @directIn;
}
