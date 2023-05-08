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

simple Ckx_car like IApp{
    parameters:
        
//        @signal[count](type="int");  
//        @statistic[coverage](title="coverage"; source="count"; record=vector);
//        
//        int localPort;
//        int destPort;
//        string destAddress = default("server");
//        string host_name = default("car");
//        int packetSize @unit(B) = default(10B);
//        
//        double period @unit(s) = default(0.2s); // selfMessage send period "first"
//        double sendInterval @unit(s) = default(0.5s);
//    	string interfaceTableModule;
//    	string multicastInterface = default("wlan");
    gates:
        output socketOut;
        input socketIn;
        input hostIn @directIn;
        output hostOut @directIn;
}
